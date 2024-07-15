//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T R I N G S
//
//   Artificial Intelligence Techniques, SL
//   artelnics artelnics.com

#ifndef STRINGS_H
#define STRINGS_H

// System includes

#include <math.h>
#include <regex>
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <string>
#include <string_view>
#include <cctype>
#include <iomanip>
#include <set>
#include <unordered_set>
#include <vector>
#include <map>
#include <numeric>
#include <tuple>

// Eigen includes

#include "config.h"
#include "tensors.h"

using namespace std;

namespace opennn
{
    Index count_tokens(const string&, const char& separator = ' ');
    Tensor<string, 1> get_tokens(const string&, const char& delimiter=' ');
    void fill_tokens(const string&, const char&, Tensor<string, 1>&);

    Index count_tokens(const string&, const string&);
    Tensor<string, 1> get_tokens(const string&, const string&);

    Tensor<type, 1> to_type_vector(const string&, const char&);
    Tensor<Index, 1> to_index_vector(const string&, const char&);

    Tensor<string, 1> get_unique_elements(const Tensor<string,1>&);
    Tensor<Index, 1> count_unique(const Tensor<string,1>&);

    bool is_numeric_string(const string&);
    bool is_date_time_string(const string&);
    bool is_email(const string&);
    bool contains_number(const string&);

    bool starts_with(const string& word, const string& starting);
    bool ends_with(const string&, const string&);
    bool ends_with(const string&, const Tensor<string,1>&);

    bool is_constant_numeric(const Tensor<type, 1>&);
    bool is_constant_string(const Tensor<string, 1>&);

    time_t date_to_timestamp(const string& date, const Index& gmt = 0);

    bool contains_substring(const string&, const string&);

    void replace_all_appearances(string& s, string const& toReplace, string const& replaceWith);
    void replace_all_word_appearances(string& s, string const& toReplace, string const& replaceWith);

    vector<string> get_words_in_a_string(string str);
    string replace_non_allowed_programming_expressions(string& s);

    Tensor<string, 1> fix_write_expression_outputs(const string&, const Tensor<string, 1>&, const string&);
    Tensor<Tensor<string,1>, 1> fix_input_output_variables(Tensor<string, 1>&, Tensor<string, 1>&, ostringstream&);

    int WordOccurrence(char *sentence, char *word);

    void trim(string&);
    void erase(string&, const char&);
    //void replace_first_and_last_char_with_missing_label(string &str, char target_char, const string &missing_label);
    void replace_first_and_last_char_with_missing_label(string &str, char target_char, const string &first_missing_label, const string &last_missing_label);

    string get_trimmed(const string&);

    string prepend(const string&, const string&);

    bool has_numbers(const Tensor<string, 1>&);
    bool has_strings(const Tensor<string, 1>&);

    bool is_numeric_string_vector(const Tensor<string, 1>&);

    bool is_not_numeric(const Tensor<string, 1>&);
    bool is_mixed(const Tensor<string, 1>&);

    void remove_non_printable_chars( string&);

    void replace(string&, const string&, const string&);
    void replace_substring(Tensor<string, 1>&, const string& , const string&);
    void replace_double_char_with_label(string&, const string&, const string&);
    void replac_substring_within_quotes(string&, const string&, const string&);
    void replace_substring_in_string (Tensor<string, 1>& found_tokens, string& outputs_espresion, const string& keyword);

    void display_progress_bar(int completed, int total);

    bool isNotAlnum(char &c);
    void remove_not_alnum(string &str);

    bool find_string_in_tensor(Tensor<string, 1>& t, string val);
    string get_word_from_token(string&);

    string round_to_precision_string(type, const int&);
    Tensor<string,2> round_to_precision_string_matrix(Tensor<type,2>, const int&);

    Tensor<string,1> sort_string_tensor (Tensor<string, 1> tensor);
    Tensor<string,1> concatenate_string_tensors (const Tensor<string, 1>& tensor_1, const Tensor<string, 1>& tensor_2);

    void print();

    void create_alphabet();

    void encode_alphabet();

    void preprocess();

    Tensor<type, 1> one_hot_encode(const string&);

    Tensor<type, 2> multiple_one_hot_encode(const string&);

    string one_hot_decode(const Tensor<type, 1>&);

    string multiple_one_hot_decode(const Tensor<type, 2>&);

    Tensor<type, 2> str_to_input(const string&);

    // Preprocess methods

    Index count(const Tensor<Tensor<string, 1>, 1>& documents);
    Tensor<string, 1> join(const Tensor<Tensor<string, 1>, 1>&);
    void to_lower(Tensor<string, 1>& documents);
    void split_punctuation(Tensor<string, 1>&);
    void delete_non_printable_chars(Tensor<string, 1>&);
    void delete_extra_spaces(Tensor<string, 1>&);
    void aux_remove_non_printable_chars(Tensor<string, 1>&);
    Tensor<Tensor<string, 1>, 1> tokenize(const Tensor<string, 1>&);
    void delete_emails(Tensor<Tensor<string, 1>, 1>&);
    void delete_blanks(Tensor<string, 1>&);
    void delete_blanks(Tensor<Tensor<string, 1>, 1>&);

    Tensor<Tensor<string, 1>, 1> preprocess_language_documents(const Tensor<string, 1>&);

    vector<pair<string, int>> count_words(const Tensor<string, 1>&);
}

#endif // OPENNNSTRINGS_H

/*

//   OpenNN: Open Neural Networks Library
//   www.opennn.net

//   T E X T   A N A L Y T I C S   C L A S S   H E A D E R

//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#ifndef __TEXTANALYTICS_H__
#define __TEXTANALYTICS_H__

// System includes

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

// OpenNN includes

#include "config.h"
#include "word_bag.h"

#include "tensors.h"
#include "strings_utilities.h"

// TinyXml includes

#include "tinyxml2.h"

namespace opennn
{

class TextGenerationAlphabet;

/// This class represent the text analytics methodata_set.
/// Text analytics is used to transform unstructured text into high-quality information.

class TextAnalytics
{

public:

    // DEFAULT CONSTRUCTOR

    explicit TextAnalytics();

    virtual ~TextAnalytics();

    /// Enumeration of available languages.

    enum Language {ENG, SPA};

    // Get methods

    Language get_language() const;

    string get_language_string() const;

    Index get_short_words_length() const;

    Index get_long_words_length() const;

    Tensor<Tensor<string, 1>,1> get_documents() const;

    Tensor<Tensor<string, 1>, 1> get_targets() const;

    Tensor<string, 1> get_stop_words() const;

    Index get_document_sentences_number() const;

    Tensor<Index, 1> get_words_number(const Tensor<Tensor<string, 1>, 1>&) const;

    Tensor<Index, 1> get_sentences_number(const Tensor<string, 1>& documents) const;

    // Set methods

    void set_language(const Language&);

    void set_language(const string&);

    void set_stop_words(const Tensor<string ,1>&);

    void set_short_words_length(const Index&);

    void set_long_words_length(const Index&);

    void set_separator(const string&);

    // Auxiliar methods

    string calculate_text_outputs(TextGenerationAlphabet&, const string&, const Index&, const bool&);

    string generate_word(TextGenerationAlphabet&, const string&, const Index&);

    string generate_phrase(TextGenerationAlphabet&, const string&, const Index&);

    void append_document(const string&);

    void append_documents(const Tensor<string, 1>&);

    void filter_not_equal_to(Tensor<string,1>&, const Tensor<string,1>&) const;

    void load_documents(const string&);

    string to_string(Tensor<string,1> token) const;

    Tensor<Tensor<string, 1>, 1> tokenize(const Tensor<string, 1>&) const;

    Tensor<string, 1> detokenize(const Tensor<Tensor<string, 1>, 1>&) const;

    Index count(const Tensor<Tensor<string, 1>, 1>&) const;

    Index calculate_weight(const Tensor<string, 1>& document_words, const WordBag& word_bag) const;

    Tensor<string, 1> join(const Tensor<Tensor<string, 1>, 1>&) const;

    string read_txt_file(const string&) const;

    // Preprocess methods

    void delete_extra_spaces(Tensor<string, 1>&) const;

    void delete_breaks_and_tabs(Tensor<string, 1>&) const;

    void delete_non_printable_chars(Tensor<string, 1>&) const;

    void delete_punctuation(Tensor<string, 1>&) const;

    void split_punctuation(Tensor<string, 1>&) const;

    void delete_stop_words(Tensor<Tensor<string, 1>, 1>&) const;

    void delete_short_words(Tensor<Tensor<string, 1>, 1>&, const Index& = 2) const;

    void delete_long_words(Tensor<Tensor<string, 1>, 1>&, const Index& = 15) const;

    void delete_numbers(Tensor<Tensor<string, 1>, 1>&) const;

    void delete_emails(Tensor<Tensor<string, 1>, 1>&) const;

    void delete_words(Tensor<Tensor<string, 1>, 1>&, const Tensor<string, 1>&) const;

    void delete_blanks(Tensor<string, 1>&) const;

    void delete_blanks(Tensor<Tensor<string, 1>, 1>&) const;

    void replace_accented(Tensor<Tensor<string, 1>, 1>&) const;

    void replace_accented(string&) const;

    void to_lower(Tensor<string, 1>&) const;

    void aux_remove_non_printable_chars(Tensor<string,1>&) const;

    // Stemming methods

    /// Reduces inflected(or sometimes derived) words to their word stem, base or root form

    string get_rv(const string&, const Tensor<string, 1>&) const;

    Tensor<string, 1> get_r1_r2(const string&, const Tensor<string, 1>&) const;

    Tensor<Tensor<string, 1>, 1> apply_stemmer(const Tensor<Tensor<string, 1>, 1>&) const;

    Tensor<Tensor<string, 1>, 1> apply_english_stemmer(const Tensor<Tensor<string, 1>, 1>&) const;

    Tensor<Tensor<string, 1>, 1> apply_spanish_stemmer(const Tensor<Tensor<string, 1>, 1>&) const;

    // Word bag

    /// It is a simplifying representation where a text(such as a sentence or a document) is represented
    /// as the bag(multiset) of its words, disregarding grammar and even word order but keeping multiplicity.

    WordBag calculate_word_bag(const Tensor<Tensor<string, 1>, 1>&) const;

    WordBag calculate_word_bag_minimum_frequency(const Tensor<Tensor<string, 1>, 1>&, const Index&) const;

    WordBag calculate_word_bag_minimum_percentage(const Tensor<Tensor<string, 1>, 1>&, const double&) const;

    WordBag calculate_word_bag_minimum_ratio(const Tensor<Tensor<string, 1>, 1>&, const double&) const;

    WordBag calculate_word_bag_total_frequency(const Tensor<Tensor<string, 1>, 1>&, const Index&) const;

    WordBag calculate_word_bag_maximum_size(const Tensor<Tensor<string, 1>, 1>&, const Index&) const;

    // Algorithms

    Tensor<Tensor<string, 1>, 1> preprocess(const Tensor<string, 1>&) const;

    Tensor<Tensor<string, 1>, 1> preprocess_language_model(const Tensor<string, 1>&) const;

    Tensor<double, 1> get_words_presence_percentage(const Tensor<Tensor<string, 1>, 1>&, const Tensor<string, 1>&) const;

    Tensor<string, 2> calculate_combinated_words_frequency(const Tensor<Tensor<string, 1>, 1>&, const Index&, const Index&) const;

    Tensor<string, 2> top_words_correlations(const Tensor<Tensor<string, 1>, 1>&, const double&, const Tensor<Index, 1>&) const;


private:

    void set_english_stop_words();
    void set_spanish_stop_words();
    void clear_stop_words();

    // MEMBERS

    /// Documents language.

    Language lang = ENG;

    /// Words which are filtered out before or after processing of natural language data.

    Tensor<string, 1> stop_words;

    Index short_words_length = 2;

    Index long_words_length = 15;

    string separator = "\t";

    Tensor<Tensor<string,1>,1> documents;

    Tensor<Tensor<string,1>,1> targets;

};


class TextGenerationAlphabet
{
public:

    // DEFAULT CONSTRUCTOR

    explicit TextGenerationAlphabet();

    explicit TextGenerationAlphabet(const string&);

    virtual ~TextGenerationAlphabet();

    // Get methods

    string get_text() const;

    Tensor<type, 2> get_data_tensor() const;

    Tensor<string, 1> get_alphabet() const;

    Index get_alphabet_length() const;

    Index get_alphabet_index(const char&) const;

    // Set methods

    void set();

    void set_text(const string&);

    void set_data_tensor(const Tensor<type, 2>&);

    void set_alphabet(const Tensor<string, 1>&);

    // Other methods

    void print() const;

    void create_alphabet();

    void encode_alphabet();

    void preprocess();

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
