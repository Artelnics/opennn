//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T O K E N I Z E R   O P E R A T O R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <array>
#include <unordered_set>

#include "operator.h"
#include "string_utilities.h"

namespace opennn
{

vector<string> make_vocabulary(const unordered_map<string_view, size_t>&,
                               span<const string> reserved_tokens,
                               Index maximum_size,
                               Index minimum_frequency);

class TokenizerOperator : public Operator
{
public:
    using VocabularyMap = StringMap<Index>;

    virtual vector<string> tokenize(string_view text) const = 0;

    virtual void build_vocabulary(const vector<vector<string>>& documents,
                                  Index maximum_vocabulary_size,
                                  Index minimum_token_frequency);

    virtual void set_vocabulary(const vector<string>&);
    const vector<string>& get_vocabulary() const noexcept { return vocabulary; }
    const VocabularyMap& get_vocabulary_map() const noexcept { return vocabulary_map; }
    Index get_vocabulary_size() const noexcept { return Index(vocabulary.size()); }

    Index token_to_id(string_view token) const;
    const string& id_to_token(Index id) const;

    // Word-level reserved-token ids; [PAD] = 0 is implicit in zero-initialized
    // buffers. Other tokenizers do not add start/end framing unless configured.
    static constexpr Index UNK_INDEX   = 1;
    static constexpr Index START_INDEX = 2;
    static constexpr Index END_INDEX   = 3;

    virtual vector<Index> encode(string_view text) const;
    vector<Index> encode_sequence(const vector<string>& tokens, Index sequence_length) const;
    vector<Index> encode_sequence(string_view text, Index sequence_length) const;
    virtual string decode(const vector<Index>& ids) const;
    virtual string decode_token(Index id) const;
    // True only when decode(ids) equals concatenating decode_token over ids,
    // so streaming consumers may decode one token at a time.
    virtual bool supports_incremental_decode() const noexcept { return false; }

    Index get_unk_id() const noexcept { return unk_id; }
    const vector<string>& get_reserved_tokens() const noexcept { return reserved_tokens; }

    virtual unique_ptr<TokenizerOperator> clone() const = 0;
    virtual string_view get_kind() const = 0;
    virtual uint64_t fingerprint() const;

    void to_JSON(JsonWriter&) const override;
    void from_JSON(const Json*) override;

protected:

    vector<string> reserved_tokens;
    Index unk_id = 0;
    Index start_id = -1;
    Index end_id = -1;

    vector<string> vocabulary;
    VocabularyMap vocabulary_map;

    void rebuild_map();
};

unique_ptr<TokenizerOperator> make_tokenizer_operator(string_view kind);

// Reserved tokens: [PAD]=0, [UNK]=1, [START]=2, [END]=3.
class WordLevelTokenizer : public TokenizerOperator
{
public:

    WordLevelTokenizer();
    explicit WordLevelTokenizer(vector<string> reserved_tokens);

    vector<string> tokenize(string_view text) const override;
    vector<Index> encode(string_view text) const override;

    unique_ptr<TokenizerOperator> clone() const override { return make_unique<WordLevelTokenizer>(*this); }
    string_view get_kind() const override { return "WordLevel"; }
};


class WordPieceTokenizer : public TokenizerOperator
{
public:

    WordPieceTokenizer();
    explicit WordPieceTokenizer(const vector<string>& vocabulary);

    void load_vocabulary(const filesystem::path& vocabulary_file);

    void set_vocabulary(const vector<string>&) override;

    vector<string> tokenize(string_view text) const override;
    vector<Index> encode(string_view text) const override;

    void build_vocabulary(const vector<vector<string>>&, Index, Index) override {}

    void set_lower_case(bool value) noexcept { do_lower_case = value; }
    bool get_lower_case() const noexcept { return do_lower_case; }

    unique_ptr<TokenizerOperator> clone() const override { return make_unique<WordPieceTokenizer>(*this); }
    string_view get_kind() const override { return "WordPiece"; }
    uint64_t fingerprint() const override;

    void to_JSON(JsonWriter&) const override;
    void from_JSON(const Json*) override;

private:

    vector<string> basic_tokenize(string_view text) const;
    void wordpiece(const string&, vector<string>*, vector<Index>*) const;

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
    BytePairTokenizer(const filesystem::path& vocabulary_json,
                      const filesystem::path& merges_txt);

    void load(const filesystem::path& vocabulary_json,
              const filesystem::path& merges_txt);

    void set_vocabulary(const vector<string>&) override;

    vector<string> tokenize(string_view text) const override;
    vector<Index> encode(string_view text) const override;
    string decode(const vector<Index>& ids) const override;
    string decode_token(Index id) const override;
    bool supports_incremental_decode() const noexcept override { return true; }

    void build_vocabulary(const vector<vector<string>>&, Index, Index) override {}

    vector<string> get_merges() const;
    void set_merges(const vector<string>&);
    void set_special_tokens(const vector<string>&);
    bool is_special(Index id) const { return special_ids.contains(id); }
    Index get_special_token_id(string_view) const;

    unique_ptr<TokenizerOperator> clone() const override { return make_unique<BytePairTokenizer>(*this); }
    string_view get_kind() const override { return "BytePair"; }
    uint64_t fingerprint() const override;

    void to_JSON(JsonWriter&) const override;
    void from_JSON(const Json*) override;

    static constexpr string_view PAD_TOKEN = "[PAD]";

protected:

    // Split raw text into pieces before byte-level BPE. Virtual so decoder-model
    // tokenizers (e.g. Qwen3) can supply their own pre-tokenization regex.
    virtual vector<string> pre_tokenize(string_view text) const;
    vector<string> bpe(const string& token) const;
    void tokenize_into(string_view, vector<string>*, vector<Index>*) const;

    array<uint32_t, 256> byte_encoder{};
    unordered_map<uint32_t, unsigned char> byte_decoder;
    StringMap<int> merge_ranks;
    // Identifies the current merge table; bumped by set_merges so the
    // thread-local BPE memoization cache in tokenize_into stays coherent.
    uint64_t merges_revision = 0;
    vector<string> special_strings;
    unordered_set<Index> special_ids;
};


// Byte-level BPE tokenizer for Qwen3 / Qwen2 decoder models. Differs from the
// GPT-2 BytePairTokenizer in two ways: (1) the Qwen pre-tokenization regex
// (each digit is its own piece; a single non-letter char can lead a word), and
// (2) ChatML special tokens (<|im_start|>, <|im_end|>, <|endoftext|>, ...) are
// matched atomically before BPE. Loads vocab.json + merges.txt + a special-token
// list. Ids are shifted +1 (id 0 = [PAD]) to match OpenNN's Embedding.
class Qwen3Tokenizer : public BytePairTokenizer
{
public:

    Qwen3Tokenizer() = default;
    explicit Qwen3Tokenizer(const filesystem::path& directory);

    // special_tokens_tsv: one "id<TAB>token" line per special/added token.
    void load(const filesystem::path& vocabulary_json,
              const filesystem::path& merges_txt,
              const filesystem::path& special_tokens_tsv);

    // ChatML / generation ids (already +1 shifted). -1 if absent.
    Index get_im_start_id()  const { return get_special_token_id("<|im_start|>"); }
    Index get_im_end_id()    const { return get_special_token_id("<|im_end|>"); }
    Index get_endoftext_id() const { return get_special_token_id("<|endoftext|>"); }

    unique_ptr<TokenizerOperator> clone() const override { return make_unique<Qwen3Tokenizer>(*this); }
    string_view get_kind() const override { return "Qwen3"; }

protected:

    vector<string> pre_tokenize(string_view text) const override;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
