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

#include "tensor_utilities.h"

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
                cout << words(i) << ":frequency= " << frequencies(i) << ", percentage= " << percentages(i) << endl;
        }
    };

    // Get methods

    Language get_language() const;
    string get_language_string() const;

    Tensor<string, 1> get_documents() const;
    Tensor<string, 1> get_stop_words() const;

    // Set methods

    void set_language(const Language&);
    void set_language(const string&);

    void set_stop_words(const Tensor<string ,1>&);

    // Auxiliar methods

    void append_document(const string&);
    void append_documents(const Tensor<string, 1>&);

    // Preprocess methods

    void load_documents(const string&);

    Tensor<string ,1> delete_extra_spaces(const Tensor<string, 1>&) const;
    Tensor<string ,1> delete_punctuation(const Tensor<string, 1>&) const;

    Tensor<string ,1> to_lower(const Tensor<string, 1>&) const;

    Tensor<Tensor<string, 1>, 1> tokenize(const Tensor<string, 1>&) const;
    Tensor<string, 1> detokenize(const Tensor<Tensor<string, 1>, 1>&) const;

    Tensor<Tensor<string, 1>, 1> delete_words(const Tensor<Tensor<string, 1>, 1>&, const Tensor<string, 1>&) const;

    Tensor<Tensor<string,1>, 1> delete_stop_words(const Tensor<Tensor<string, 1>, 1>&) const;

    Tensor<Tensor<string,1>, 1> delete_short_words(const Tensor<Tensor<string, 1>, 1>&, const Index& = 2) const;

    Tensor<Tensor<string,1>, 1> delete_long_words(const Tensor<Tensor<string, 1>, 1>&, const Index& = 15) const;

    Tensor<Tensor<string,1>, 1> delete_numbers(const Tensor<Tensor<string, 1>, 1>&) const;

    Tensor<Tensor<string,1>, 1> delete_emails(const Tensor<Tensor<string, 1>, 1>&) const;

    Tensor<Tensor<string,1>, 1> replace_accented(const Tensor<Tensor<string, 1>, 1>&) const;

    // Stemming methods

    /// Reduces inflected(or sometimes derived) words to their word stem, base or root form

    Tensor<string, 1> get_r1_r2(const string&, const Tensor<string, 1>&) const;
    string get_rv(const string&, const Tensor<string, 1>&) const;

    string replace_accented(const string&) const;

    Tensor<Tensor<string, 1>, 1> apply_stemmer(const Tensor<Tensor<string, 1>, 1>&) const;
    Tensor<Tensor<string, 1>, 1> apply_english_stemmer(const Tensor<Tensor<string, 1>, 1>&) const;
    Tensor<Tensor<string, 1>, 1> apply_spanish_stemmer(const Tensor<Tensor<string, 1>, 1>&) const;

    // Word bag

    /// It is a simplifying representation where a text(such as a sentence or a document) is represented
    /// as the bag(multiset) of its words, disregarding grammar and even word order but keeping multiplicity.

    Index count(const Tensor<Tensor<string, 1>, 1>&) const;

    Tensor<string, 1> join(const Tensor<Tensor<string, 1>, 1>&) const;

    WordBag calculate_word_bag(const Tensor<Tensor<string, 1>, 1>&) const;

    WordBag calculate_word_bag_minimum_frequency(const Tensor<Tensor<string, 1>, 1>&, const Index&) const;
    WordBag calculate_word_bag_minimum_percentage(const Tensor<Tensor<string, 1>, 1>&, const double&) const;
    WordBag calculate_word_bag_minimum_ratio(const Tensor<Tensor<string, 1>, 1>&, const double&) const;
    WordBag calculate_word_bag_total_frequency(const Tensor<Tensor<string, 1>, 1>&, const Index&) const;
    WordBag calculate_word_bag_maximum_size(const Tensor<Tensor<string, 1>, 1>&, const Index&) const;

    Index calculate_weight(const Tensor<string, 1>&, const TextAnalytics::WordBag&) const;

    // Algorithms

    Tensor<Tensor<string, 1>, 1> preprocess(const Tensor<string, 1>&) const;

    Tensor<Index, 1> get_words_number(const Tensor<Tensor<string, 1>, 1>&) const;

    Tensor<Index, 1> get_sentences_number(const Tensor<string, 1>& documents) const;

    Tensor<double, 1> get_words_presence_percentage(const Tensor<Tensor<string, 1>, 1>&, const Tensor<string, 1>&) const;

    Tensor<string, 2> calculate_combinated_words_frequency(const Tensor<Tensor<string, 1>, 1>&, const Index&, const Index&) const;

    Tensor<string, 2> top_words_correlations(const Tensor<Tensor<string, 1>, 1>&, const double&, const Tensor<Index, 1>&) const;

    Tensor<double, 2> calculate_data_set(const Tensor<string, 1>&, const Tensor<string, 1>&, const TextAnalytics::WordBag&) const;

    // Binarize methods

    Tensor<double, 1> get_binary_Tensor(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

    Tensor<double, 2> get_binary_matrix(const Tensor<string, 1>&, const char& separator = ' ') const;

    Tensor<double, 2> get_unique_binary_matrix(const Tensor<string, 1>&, const char&, const Tensor<string, 1>&) const;

private:

    void set_english_stop_words();
    void set_spanish_stop_words();
    void clear_stop_words();

    bool is_number(const string&) const;
    bool contains_number(const string&) const;

    bool is_email(const string&) const;

    bool starts_with(const string&, const string&) const;

    bool ends_with(const string&, const string&) const;
    bool ends_with(const string&, const Tensor<string, 1>&) const;

    // MEMBERS

    /// Documents language.

    Language lang = ENG;

    /// Words which are filtered out before or after processing of natural language data.

    Tensor<string, 1> stop_words;

    Tensor<string, 1> punctuation_characters;
};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2019 Artificial Intelligence Techniques, SL.
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

