//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T R I N G S
//
//   Artificial Intelligence Techniques, SL
//   artelnics artelnics.com

#ifndef STRINGS_H
#define STRINGS_H

#include "pch.h"

namespace opennn
{
    void fill_tokens(const string&, const string&, vector<string>&);

    Index count_tokens(const string&, const string&);
    vector<string> get_tokens(const string&, const string&);

    Tensor<type, 1> to_type_vector(const string&, const string&);
    Tensor<Index, 1> to_index_vector(const string&, const string&);

    vector<string> get_unique_elements(const vector<string>&);
    Tensor<Index, 1> count_unique(const vector<string>&);

    bool is_numeric_string(const string&);
    bool is_date_time_string(const string&);
    bool is_email(const string&);

    bool starts_with(const string&, const string&);
    bool ends_with(const string&, const string&);
    bool ends_with(const string&, const vector<string>&);

    time_t date_to_timestamp(const string&, const Index& = 0);

    bool contains_substring(const string&, const string&);

    void replace_all_appearances(string&, const string&, const string&);
    void replace_all_word_appearances(string&, const string&, const string&);

    vector<string> get_words_in_a_string(const string&);
    string replace_non_allowed_programming_expressions(string&);

    vector<string> fix_get_expression_outputs(const string&, const vector<string>&, const string&);
    vector<vector<string>> fix_input_output_variables(vector<string>&, vector<string>&, ostringstream&);

    void trim(string&);
    void erase(string&, const char&);

    void replace_first_and_last_char_with_missing_label(string&, char, const string&, const string&);

    string get_trimmed(const string&);

    string prepend(const string&, const string&);

    bool has_numbers(const vector<string>&);

    bool is_numeric_string_vector(const vector<string>&);

    bool is_not_numeric(const vector<string>&);
    bool is_mixed(const vector<string>&);

    void delete_non_printable_chars(string&);

    void replace(string&, const string&, const string&);
    void replace_substring(vector<string>&, const string& , const string&);
    void replace_double_char_with_label(string&, const string&, const string&);
    void replace_substring_within_quotes(string&, const string&, const string&);
    void replace_substring_in_string (vector<string>&, string&, const string&);

    void display_progress_bar(const int&, const int&);

    bool is_not_alnum(char &c);

    bool contains(vector<string>&, const string&);
    string get_first_word(string&);

    vector<string> sort_string_vector(vector<string>&);
    vector<string> concatenate_string_vectors (const vector<string>&, const vector<string>&);

    Index count_tokens(const vector<vector<string>>&);
    vector<string> tokens_list(const vector<vector<string>>&);
    void to_lower(string&);
    void to_lower(vector<string>&);
    void split_punctuation(vector<string>&);
    void delete_non_printable_chars(vector<string>&);
    void delete_extra_spaces(vector<string>&);
    void delete_non_alphanumeric(vector<string>&);
    vector<vector<string>> get_tokens(const vector<string>&, const string&);
    void delete_blanks(vector<string>&);
    void delete_blanks(vector<vector<string>>&);

    vector<vector<string>> preprocess_language_documents(const vector<string>&);

    vector<pair<string, Index>> count_words(const vector<string>&);

    enum Language {ENG, SPA};

    Tensor<Index, 1> get_words_number(const vector<vector<string>>&);

    Tensor<Index, 1> get_sentences_number(const vector<string>&);

    void set_language(const Language&);

    void set_language(const string&);

    void set_stop_words(const Tensor<string ,1>&);

    void set_separator(const string&);

    void append_document(const string&);

    void append_documents(const vector<string>&);

    vector<string> join(const vector<vector<string>>&);

    // Preprocess

    void delete_extra_spaces(vector<string>&);

    void delete_non_printable_chars(vector<string>&);

    void split_punctuation(vector<string>&);

    void delete_emails(vector<vector<string>>&);

    void delete_blanks(vector<string>&);

    void delete_blanks(vector<vector<string>>&);

    void replace_accented_words(vector<vector<string>>&);

    void replace_accented_words(string&);

    void delete_non_alphanumeric(vector<string>&);

    vector<vector<string>> preprocess_language_model(const vector<string>&);

    bool is_vowel(char);

    bool ends_with(const string&, const string&);
}

#endif // OPENNNSTRINGS_H
