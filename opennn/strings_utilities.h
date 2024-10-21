//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T R I N G S
//
//   Artificial Intelligence Techniques, SL
//   artelnics artelnics.com

#ifndef STRINGS_H
#define STRINGS_H



#include <string>
#include <vector>

// Eigen includes

#include "config.h"
#include "word_bag.h"

using namespace std;

namespace opennn
{
//    Index count_tokens(const string&, const char& = ' ');
//    Tensor<string, 1> get_tokens(const string&, const char& =' ');
    void fill_tokens(const string&, const string&, Tensor<string, 1>&);

    Index count_tokens(const string&, const string&);
    Tensor<string, 1> get_tokens(const string&, const string&);

    Tensor<type, 1> to_type_vector(const string&, const string&);
    Tensor<Index, 1> to_index_vector(const string&, const string&);

    Tensor<string, 1> get_unique_elements(const Tensor<string,1>&);
    Tensor<Index, 1> count_unique(const Tensor<string,1>&);

    bool is_numeric_string(const string&);
    bool is_date_time_string(const string&);
    bool is_email(const string&);
//    bool contains_number(const string&);

    bool starts_with(const string&, const string&);
    bool ends_with(const string&, const string&);
    bool ends_with(const string&, const Tensor<string,1>&);

//    bool is_constant_numeric(const Tensor<type, 1>&);
//    bool is_constant_string(const Tensor<string, 1>&);

    time_t date_to_timestamp(const string&, const Index& = 0);

    bool contains_substring(const string&, const string&);

    void replace_all_appearances(string&, const string&, const string&);
    void replace_all_word_appearances(string&, const string&, const string&);

    vector<string> get_words_in_a_string(const string&);
    string replace_non_allowed_programming_expressions(string&);

    Tensor<string, 1> fix_write_expression_outputs(const string&, const Tensor<string, 1>&, const string&);
    Tensor<Tensor<string,1>, 1> fix_input_output_variables(Tensor<string, 1>&, Tensor<string, 1>&, ostringstream&);

    //int WordOccurrence(char *sentence, char *word);

    void trim(string&);
    void erase(string&, const char&);
    //void replace_first_and_last_char_with_missing_label(string &str, char target_char, const string &missing_label);
    void replace_first_and_last_char_with_missing_label(string&, char, const string&, const string&);

    string get_trimmed(const string&);

    string prepend(const string&, const string&);

    bool has_numbers(const Tensor<string, 1>&);
    bool has_strings(const Tensor<string, 1>&);

    bool is_numeric_string_vector(const Tensor<string, 1>&);

    bool is_not_numeric(const Tensor<string, 1>&);
    bool is_mixed(const Tensor<string, 1>&);

    void delete_non_printable_chars(string&);

    void replace(string&, const string&, const string&);
    void replace_substring(Tensor<string, 1>&, const string& , const string&);
    void replace_double_char_with_label(string&, const string&, const string&);
    void replac_substring_within_quotes(string&, const string&, const string&);
    void replace_substring_in_string (Tensor<string, 1>&, string&, const string&);

    void display_progress_bar(const int&, const int&);

    bool is_not_alnum(char &c);
    void remove_not_alnum(string &str);

    bool find_string_in_tensor(Tensor<string, 1>&, const string&);
    string get_word_from_token(string&);

    string round_to_precision_string(const type&, const int&);
    Tensor<string,2> round_to_precision_string_matrix(const Tensor<type,2>&, const int&);

    Tensor<string,1> sort_string_tensor(Tensor<string, 1>&);
    Tensor<string,1> concatenate_string_tensors (const Tensor<string, 1>&, const Tensor<string, 1>&);

    void print();

//    void create_alphabet();

//    void encode_alphabet();

//    Tensor<type, 1> one_hot_encode(const string&);

//    Tensor<type, 2> multiple_one_hot_encode(const string&);

//    string one_hot_decode(const Tensor<type, 1>&);

//    string multiple_one_hot_decode(const Tensor<type, 2>&);

    //Tensor<type, 2> str_to_input(const string&);

    // Preprocess

    Index count_tokens(const Tensor<Tensor<string, 1>, 1>&);
    Tensor<string, 1> tokens_list(const Tensor<Tensor<string, 1>, 1>&);
    void to_lower(string&);
    void to_lower(Tensor<string, 1>&);
//    void to_lower(Tensor<Tensor<string, 1>, 1>&);
    void split_punctuation(Tensor<string, 1>&);
    void delete_non_printable_chars(Tensor<string, 1>&);
    void delete_extra_spaces(Tensor<string, 1>&);
    void delete_non_alphanumeric(Tensor<string, 1>&);
    Tensor<Tensor<string, 1>, 1> get_tokens(const Tensor<string, 1>&, const string&);
    void delete_blanks(Tensor<string, 1>&);
    void delete_blanks(Tensor<Tensor<string, 1>, 1>&);

    Tensor<Tensor<string, 1>, 1> preprocess_language_documents(const Tensor<string, 1>&);

    vector<pair<string, int>> count_words(const Tensor<string, 1>&);

    enum Language {ENG, SPA};

    // Get

    Language get_language();

    string get_language_string();

    Index get_short_words_length();

    Index get_long_words_length();

    Tensor<Tensor<string, 1>,1> get_documents();

    Tensor<Tensor<string, 1>, 1> get_targets();

    Tensor<string, 1> get_stop_words();

    Tensor<Index, 1> get_words_number(const Tensor<Tensor<string, 1>, 1>&);

    Tensor<Index, 1> get_sentences_number(const Tensor<string, 1>&);

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

    void append_documents(const Tensor<string, 1>&);

    void filter_not_equal_to(Tensor<string,1>&, const Tensor<string,1>&);

    //Tensor<Tensor<string, 1>, 1> get_tokens(const Tensor<string, 1>&);

    Tensor<string, 1> detokenize(const Tensor<Tensor<string, 1>, 1>&);

    Index count(const Tensor<Tensor<string, 1>, 1>&);

    Index calculate_weight(const Tensor<string, 1>&, const WordBag&);

    Tensor<string, 1> join(const Tensor<Tensor<string, 1>, 1>&);

    // Preprocess

    void delete_extra_spaces(Tensor<string, 1>&);

//    void delete_breaks_and_tabs(Tensor<string, 1>&);

    void delete_non_printable_chars(Tensor<string, 1>&);

    void delete_punctuation(Tensor<string, 1>&);

    void split_punctuation(Tensor<string, 1>&);

    void delete_short_long_words(Tensor<Tensor<string, 1>, 1>&, const Index& = 2, const Index& = 15);

    void delete_numbers(Tensor<Tensor<string, 1>, 1>&);

    void delete_emails(Tensor<Tensor<string, 1>, 1>&);

    void delete_words(Tensor<Tensor<string, 1>, 1>&, const Tensor<string, 1>&);

    void delete_blanks(Tensor<string, 1>&);

    void delete_blanks(Tensor<Tensor<string, 1>, 1>&);

    void replace_accented_words(Tensor<Tensor<string, 1>, 1>&);

    void replace_accented_words(string&);

    void delete_non_alphanumeric(Tensor<string,1>&);

    // Stemming

    string get_rv(const string&, const Tensor<string, 1>&);

    Tensor<string, 1> get_r1_r2(const string&, const Tensor<string, 1>&);

    // Word bag

    WordBag calculate_word_bag(const Tensor<string, 1>&);

//    WordBag calculate_word_bag_minimum_frequency(const Tensor<Tensor<string, 1>, 1>&, const Index&);

//    WordBag calculate_word_bag_minimum_percentage(const Tensor<Tensor<string, 1>, 1>&, const double&);

//    WordBag calculate_word_bag_minimum_ratio(const Tensor<Tensor<string, 1>, 1>&, const double&);

//    WordBag calculate_word_bag_total_frequency(const Tensor<Tensor<string, 1>, 1>&, const Index&);

//    WordBag calculate_word_bag_maximum_size(const Tensor<Tensor<string, 1>, 1>&, const Index&);

    // Algorithms

    Tensor<Tensor<string, 1>, 1> preprocess(const Tensor<string, 1>&);

    Tensor<Tensor<string, 1>, 1> preprocess_language_model(const Tensor<string, 1>&);

//    Tensor<double, 1> get_words_presence_percentage(const Tensor<Tensor<string, 1>, 1>&, const Tensor<string, 1>&);

//    Tensor<string, 2> calculate_combinated_words_frequency(const Tensor<Tensor<string, 1>, 1>&, const Index&, const Index&);

//    Tensor<string, 2> top_words_correlations(const Tensor<Tensor<string, 1>, 1>&, const double&, const Tensor<Index, 1>&);

    bool is_vowel(char);

    bool ends_with(const string&, const string&);

    int measure(const string&);

    bool contains_vowel(const string&);

    bool is_double_consonant(const string&);

    bool is_consonant_vowel_consonant(const string&);

    string stem(const string&);
    void stem(Tensor<string, 1>&);
    void stem(Tensor<Tensor<string, 1>, 1>&);

//    void print_tokens(const Tensor<Tensor<string,1>,1>&);
}

#endif // OPENNNSTRINGS_H

/*

class TextGenerationAlphabet
{
public:

    // DEFAULT CONSTRUCTOR

    explicit TextGenerationAlphabet();

    explicit TextGenerationAlphabet(const string&);

    virtual ~TextGenerationAlphabet();

    // Get

    string get_text() const;

    Tensor<type, 2> get_data_tensor() const;

    Tensor<string, 1> get_alphabet() const;

    Index get_alphabet_length() const;

    Index get_alphabet_index(const char&) const;

    // Set

    void set();

    void set_text(const string&);

    void set_data_tensor(const Tensor<type, 2>&);

    void set_alphabet(const Tensor<string, 1>&);

    // Other

    void print() const;

    void create_alphabet();

    void encode_alphabet();

    Tensor<type, 1> one_hot_encode(const string &) const;

    Tensor<type, 2> multiple_one_hot_encode(const string &) const;

    string one_hot_decode(const Tensor<type, 1>&) const;

    string multiple_one_hot_decode(const Tensor<type, 2>&) const;

    Tensor<type, 2> str_to_input(const string &) const;

private:

    string text;

    Tensor<type, 2> data_tensor;

    Tensor<string, 1> alphabet;

};

}

#endif

*/
