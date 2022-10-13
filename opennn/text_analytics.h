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
#include "tensor_utilities.h"
#include "opennn_strings.h"

// TinyXml includes

#include "tinyxml2.h"

namespace opennn
{

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

    ///
    /// This structure is a necessary tool in text analytics, the word bag is similar a notebook
    /// where you store the words and statistical processing is done to obtain relevant information.
    /// Return various list with words, repetition frequencies and percentages.

    struct WordBag
    {
        /// Default constructor.

        explicit WordBag() {}

        /// Destructor.

        virtual ~WordBag() {}

        Tensor<string, 1> words;
        Tensor<Index, 1> frequencies;
        Tensor<double, 1> percentages;

        Index size() const
        {
            return words.size();
        }

        void print() const
        {
            const Index words_size = words.size();

            cout << "Word bag size: " << words_size << endl;

            for(Index i = 0; i < words_size; i++)
                cout << words(i) << ": frequency= " << frequencies(i) << ", percentage= " << percentages(i) << endl;
        }
    };

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

    void append_document(const string&);

    void append_documents(const Tensor<string, 1>&);

    void filter_not_equal_to(Tensor<string,1>&, const Tensor<string,1>&) const;

    void load_documents(const string&);

    string to_string(Tensor<string,1> token) const;

    Tensor<Tensor<string, 1>, 1> tokenize(const Tensor<string, 1>&) const;

    Tensor<string, 1> detokenize(const Tensor<Tensor<string, 1>, 1>&) const;

    Index count(const Tensor<Tensor<string, 1>, 1>&) const;

    Tensor<string, 1> join(const Tensor<Tensor<string, 1>, 1>&) const;

    string read_txt_file(const string&) const;

    // Preprocess methods

    void delete_extra_spaces(Tensor<string, 1>&) const;

    void delete_breaks_and_tabs(Tensor<string, 1>&) const;

    void delete_non_printable_chars(Tensor<string, 1>&) const;

    void delete_punctuation(Tensor<string, 1>&) const;

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

    Index calculate_weight(const Tensor<string, 1>&, const TextAnalytics::WordBag&) const;

    // Algorithms

    Tensor<Tensor<string, 1>, 1> preprocess(const Tensor<string, 1>&) const;

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

private:

    string text;

    Tensor<type, 2> data_tensor;

    Tensor<string, 1> alphabet;

};

}



#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software

// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

