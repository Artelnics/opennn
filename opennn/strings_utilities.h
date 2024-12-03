//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T R I N G S
//
//   Artificial Intelligence Techniques, SL
//   artelnics artelnics.com

#ifndef STRINGS_H
#define STRINGS_H

#include "word_bag.h"

namespace opennn
{
//    Index count_tokens(const string&, const char& = ' ');
//    vector<string> get_tokens(const string&, const char& =' ');
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
//    bool contains_number(const string&);

    bool starts_with(const string&, const string&);
    bool ends_with(const string&, const string&);
    bool ends_with(const string&, const vector<string>&);

//    bool is_constant_numeric(const Tensor<type, 1>&);
//    bool is_constant_string(const vector<string>&);

    time_t date_to_timestamp(const string&, const Index& = 0);

    bool contains_substring(const string&, const string&);

    void replace_all_appearances(string&, const string&, const string&);
    void replace_all_word_appearances(string&, const string&, const string&);

    vector<string> get_words_in_a_string(const string&);
    string replace_non_allowed_programming_expressions(string&);

    vector<string> fix_get_expression_outputs(const string&, const vector<string>&, const string&);
    vector<vector<string>> fix_input_output_variables(vector<string>&, vector<string>&, ostringstream&);

    //int WordOccurrence(char *sentence, char *word);

    void trim(string&);
    void erase(string&, const char&);
    //void replace_first_and_last_char_with_missing_label(string &str, char target_char, const string &missing_label);
    void replace_first_and_last_char_with_missing_label(string&, char, const string&, const string&);

    string get_trimmed(const string&);

    string prepend(const string&, const string&);

    bool has_numbers(const vector<string>&);
//    bool has_strings(const vector<string>&);

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
    void remove_not_alnum(string &str);

    bool contains(vector<string>&, const string&);
    string get_first_word(string&);

    string round_to_precision_string(const type&, const int&);

    vector<string> sort_string_tensor(vector<string>&);
    vector<string> concatenate_string_tensors (const vector<string>&, const vector<string>&);

    void print();

//    void create_alphabet();

//    void encode_alphabet();

//    Tensor<type, 1> one_hot_encode(const string&);

//    Tensor<type, 2> multiple_one_hot_encode(const string&);

//    string one_hot_decode(const Tensor<type, 1>&);

//    string multiple_one_hot_decode(const Tensor<type, 2>&);

    // Preprocess

    Index count_tokens(const vector<vector<string>>&);
    vector<string> tokens_list(const vector<vector<string>>&);
    void to_lower(string&);
    void to_lower(vector<string>&);
//    void to_lower(vector<vector<string>>&);
    void split_punctuation(vector<string>&);
    void delete_non_printable_chars(vector<string>&);
    void delete_extra_spaces(vector<string>&);
    void delete_non_alphanumeric(vector<string>&);
    vector<vector<string>> get_tokens(const vector<string>&, const string&);
    void delete_blanks(vector<string>&);
    void delete_blanks(vector<vector<string>>&);

    vector<vector<string>> preprocess_language_documents(const vector<string>&);

    vector<pair<string, int>> count_words(const vector<string>&);

    enum Language {ENG, SPA};

    // Get

    Language get_language();

    string get_language_string();

    Index get_short_words_length();

    Index get_long_words_length();

    vector<vector<string>> get_documents();

    vector<vector<string>> get_targets();

    vector<string> get_stop_words();

    Tensor<Index, 1> get_words_number(const vector<vector<string>>&);

    Tensor<Index, 1> get_sentences_number(const vector<string>&);

    // Set

    void set_language(const Language&);

    void set_language(const string&);

    void set_stop_words(const Tensor<string ,1>&);

    void set_separator(const string&);

    // Auxiliar

    //string calculate_text_outputs(TextGenerationAlphabet&, const string&, const Index&, const bool&);

    //string generate_word(TextGenerationAlphabet&, const string&, const Index&);

    //string generate_phrase(TextGenerationAlphabet&, const string&, const Index&);

    void append_document(const string&);

    void append_documents(const vector<string>&);

    void filter_not_equal_to(vector<string>&, const vector<string>&);

    //vector<vector<string>> get_tokens(const vector<string>&);

    vector<string> detokenize(const vector<vector<string>>&);

    //Index count(const vector<vector<string>>&);

    Index calculate_weight(const vector<string>&, const WordBag&);

    vector<string> join(const vector<vector<string>>&);

    // Preprocess

    void delete_extra_spaces(vector<string>&);

//    void delete_breaks_and_tabs(vector<string>&);

    void delete_non_printable_chars(vector<string>&);

    void delete_punctuation(vector<string>&);

    void split_punctuation(vector<string>&);

    void delete_short_long_words(vector<vector<string>>&, const Index& = 2, const Index& = 15);

    void delete_numbers(vector<vector<string>>&);

    void delete_emails(vector<vector<string>>&);

    void delete_words(vector<vector<string>>&, const vector<string>&);

    void delete_blanks(vector<string>&);

    void delete_blanks(vector<vector<string>>&);

    void replace_accented_words(vector<vector<string>>&);

    void replace_accented_words(string&);

    void delete_non_alphanumeric(vector<string>&);

    // Stemming

    string get_rv(const string&, const vector<string>&);

    vector<string> get_r1_r2(const string&, const vector<string>&);

    // Word bag

    WordBag calculate_word_bag(const vector<string>&);

//    WordBag calculate_word_bag_minimum_frequency(const vector<vector<string>>&, const Index&);

//    WordBag calculate_word_bag_minimum_percentage(const vector<vector<string>>&, const double&);

//    WordBag calculate_word_bag_minimum_ratio(const vector<vector<string>>&, const double&);

//    WordBag calculate_word_bag_total_frequency(const vector<vector<string>>&, const Index&);

//    WordBag calculate_word_bag_maximum_size(const vector<vector<string>>&, const Index&);

    // Algorithms

    vector<vector<string>> preprocess(const vector<string>&);

    vector<vector<string>> preprocess_language_model(const vector<string>&);

//    Tensor<double, 1> get_words_presence_percentage(const vector<vector<string>>&, const vector<string>&);

//    Tensor<string, 2> calculate_combinated_words_frequency(const vector<vector<string>>&, const Index&, const Index&);

//    Tensor<string, 2> top_words_correlations(const vector<vector<string>>&, const double&, const Tensor<Index, 1>&);

    bool is_vowel(char);

    bool ends_with(const string&, const string&);

    int measure(const string&);

    bool contains_vowel(const string&);

    bool is_double_consonant(const string&);

    bool is_consonant_vowel_consonant(const string&);

    string stem(const string&);
    void stem(vector<string>&);
    void stem(vector<vector<string>>&);

//    void print_tokens(const vector<vector<string>>&);
}

#endif // OPENNNSTRINGS_H

/*

class TextGenerationAlphabet
{
public:

    explicit TextGenerationAlphabet();

    explicit TextGenerationAlphabet(const string&);

    virtual ~TextGenerationAlphabet();

    // Get

    string get_text() const;

    Tensor<type, 2> get_data_tensor() const;

    vector<string> get_alphabet() const;

    Index get_alphabet_length() const;

    Index get_alphabet_index(const char&) const;

    // Set

    void set_text(const string&);

    void set_data_tensor(const Tensor<type, 2>&);

    void set_alphabet(const vector<string>&);

    // Other

    void print() const;

    void create_alphabet();

    void encode_alphabet();

    Tensor<type, 1> one_hot_encode(const string &) const;

    Tensor<type, 2> multiple_one_hot_encode(const string &) const;

    string one_hot_decode(const Tensor<type, 1>&) const;

    string multiple_one_hot_decode(const Tensor<type, 2>&) const;

private:

    string text;

    Tensor<type, 2> data_tensor;

    vector<string> alphabet;

};

}

#endif

*/
