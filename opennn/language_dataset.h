//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A N G U A G E   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file language_dataset.h
 * @brief Declares the LanguageDataset specialization of Dataset for
 *        text data, including tokenization and vocabulary management.
 */

#pragma once

#include "dataset.h"

namespace opennn
{

/**
 * @class LanguageDataset
 * @brief Dataset specialization for tokenized text.
 *
 * Stores per-sample input and target token id sequences plus the
 * corresponding string vocabularies. Supports CSV loading, vocabulary
 * construction with frequency cutoffs and three encoding modes:
 * sequence-to-sequence, classification and standalone input encoding.
 *
 * Reserves four special tokens at fixed indices: PAD (0), UNK (1),
 * START (2), END (3).
 */
class LanguageDataset final : public Dataset
{

public:

    /**
     * @brief Constructs a LanguageDataset, optionally loading from CSV.
     * @param path Path to a CSV file; pass an empty path for an empty dataset.
     */
    LanguageDataset(const filesystem::path& path = "");
    /**
     * @brief Constructs an empty LanguageDataset of given dimensions.
     * @param samples_number Number of samples.
     * @param maximum_input_sequence_length Maximum input sequence length.
     * @param maximum_target_sequence_length Maximum target sequence length.
     */
    LanguageDataset(const Index samples_number,
                    Index maximum_input_sequence_length,
                    Index maximum_target_sequence_length);

    /** @brief Read-only access to the input-side vocabulary. */
    const vector<string>& get_input_vocabulary() const { return input_vocabulary; }
    /** @brief Read-only access to the target-side vocabulary. */
    const vector<string>& get_target_vocabulary() const { return target_vocabulary; }

    /** @brief Number of distinct tokens in the input vocabulary. */
    Index get_input_vocabulary_size() const { return input_vocabulary.size(); }
    /** @brief Number of distinct tokens in the target vocabulary. */
    Index get_target_vocabulary_size() const { return target_vocabulary.size(); }

    /** @brief Maximum input sequence length supported. */
    Index get_maximum_input_sequence_length() const { return maximum_input_sequence_length; }
    /** @brief Maximum target sequence length supported. */
    Index get_maximum_target_sequence_length() const { return maximum_target_sequence_length; }

    /**
     * @brief Replaces the input vocabulary.
     * @param new_vocabulary New token list (order defines token ids).
     */
    void set_input_vocabulary(const vector<string>& new_vocabulary) { input_vocabulary = new_vocabulary; }
    /**
     * @brief Replaces the target vocabulary.
     * @param new_vocabulary New token list (order defines token ids).
     */
    void set_target_vocabulary(const vector<string>& new_vocabulary) { target_vocabulary = new_vocabulary; }

    /** @brief Reads tokens from the configured CSV file into the dataset. */
    void read_csv() override;

    /**
     * @brief Builds a vocabulary from a list of tokenized samples.
     *
     * Receives the tokenized samples as a vector of token vectors, and the
     * output vector to fill with the resulting vocabulary (limited by the
     * minimum_token_frequency and maximum_vocabulary_size fields).
     */
    void create_vocabulary(const vector<vector<string>>&, vector<string>&) const;

    /**
     * @brief Encodes input tokens into ids and stores them in the dataset.
     *
     * Receives the tokenized input samples; out-of-vocabulary tokens are
     * mapped to UNK_INDEX.
     */
    void encode_input(const vector<vector<string>>&);
    /**
     * @brief Encodes decoder target tokens for sequence-to-sequence training.
     *
     * Receives the tokenized target samples; START_TOKEN and END_TOKEN are
     * inserted automatically.
     */
    void encode_decoder_target_sequence_to_sequence(const vector<vector<string>>&);
    /**
     * @brief Encodes target tokens for classification training.
     *
     * Receives the tokenized target samples; one token per sample is expected.
     */
    void encode_target_classification(const vector<vector<string>>&);

    /**
     * @brief Loads dataset metadata and vocabularies from a parsed JSON document.
     */
    void from_JSON(const JsonDocument&) override;
    /**
     * @brief Writes dataset metadata and vocabularies to a streaming JSON writer.
     */
    void to_JSON(JsonWriter&) const override;

    /** @brief Padding token text (id 0). */
    inline static const string PAD_TOKEN   = "[PAD]";
    /** @brief Unknown token text (id 1). */
    inline static const string UNK_TOKEN   = "[UNK]";
    /** @brief Start-of-sequence token text (id 2). */
    inline static const string START_TOKEN = "[START]";
    /** @brief End-of-sequence token text (id 3). */
    inline static const string END_TOKEN   = "[END]";

    /** @brief Numeric id used for unknown tokens. */
    inline static const float UNK_INDEX = 1.0f;
    /** @brief Numeric id used for the start-of-sequence token. */
    inline static const float START_INDEX = 2.0f;
    /** @brief Numeric id used for the end-of-sequence token. */
    inline static const float END_INDEX = 3.0f;

    /** @brief Special tokens reserved at the start of every vocabulary. */
    inline static const vector<string> reserved_tokens = {PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN};

private:

    /**
     * @brief Builds a token-to-index map from a vocabulary.
     * @param vocabulary Token list whose indices follow vector order.
     * @return Hash map suitable for fast token lookup during encoding.
     */
    unordered_map<string, Index> create_vocabulary_map(const vector<string>& vocabulary);

    /** @brief Input-side vocabulary. */
    vector<string> input_vocabulary;
    /** @brief Target-side vocabulary. */
    vector<string> target_vocabulary;

    /** @brief Maximum input sequence length supported. */
    Index maximum_input_sequence_length = 0;
    /** @brief Maximum target sequence length supported. */
    Index maximum_target_sequence_length = 0;

    /** @brief Tokens that appear fewer times than this are dropped from the vocabulary. */
    Index minimum_token_frequency = 1;
    /** @brief Upper bound on the vocabulary size (per side). */
    Index maximum_vocabulary_size = 20000;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
