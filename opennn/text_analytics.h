/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   T E X T   A N A L Y T I C S   C L A S S   H E A D E R                                                      */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

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
#include <regex>

// OpenNN includes

#include "vector.h"
#include "matrix.h"
#include "correlation_analysis.h"

// TinyXml includes

#include "tinyxml2.h"

namespace OpenNN
{

class TextAnalytics
{

public:

    // DEFAULT CONSTRUCTOR

    explicit TextAnalytics();

    // DESTRUCTOR

    virtual ~TextAnalytics();

    enum Language {ENG, SPA};

    struct WordBag
    {
        /// Default constructor.

        explicit WordBag() {}

        /// Destructor.

        virtual ~WordBag() {}

        Vector<string> words;
        Vector<size_t> frequencies;
        Vector<double> percentages;

        //Vector<double> percentages;

        size_t size() const
        {
            return words.size();
        }

        void print() const
        {
            cout << "Word bag size: " << words.size() << endl;

            for(size_t i = 0; i < words.size(); i++)
                cout << words[i] << ":frequency= " << frequencies[i] << ", percentage= " << percentages[i] << endl;
        }
    };

    // GET METHODS

    Language get_language() const;
    string get_language_string() const;

    Vector<string> get_documents() const;
    Vector<string> get_stop_words() const;

    // SET METHODS

    void set_language(const Language&);
    void set_language(const string&);

    void set_stop_words(const Vector<string>&);

    // Auxiliar methods

    void append_document(const string&);
    void append_documents(const Vector<string>&);

    // Preprocess methods

    void load_documents(const string&);

    Vector<string> delete_extra_spaces(const Vector<string>&) const;
    Vector<string> delete_punctuation(const Vector<string>&) const;

    Vector<string> to_lower(const Vector<string>&) const;

    Vector< Vector<string> > tokenize(const Vector<string>&) const;
    Vector<string> detokenize(const Vector< Vector<string> >&) const;

    Vector< Vector<string> > delete_words(const Vector< Vector<string> >&, const Vector<string>&) const;

    Vector< Vector<string> > delete_stop_words(const Vector< Vector<string> >&) const;

    Vector< Vector<string> > delete_short_words(const Vector< Vector<string> >&, const size_t& = 2) const;

    Vector< Vector<string> > delete_long_words(const Vector< Vector<string> >&, const size_t& = 15) const;

    Vector< Vector<string> > delete_numbers(const Vector< Vector<string> >&) const;

    Vector< Vector<string> > delete_emails(const Vector< Vector<string> >&) const;

    Vector< Vector<string> > replace_accented(const Vector< Vector<string> >&) const;

    // Stemming methods

    /// Reduces inflected(or sometimes derived) words to their word stem, base or root form

    Vector<string> get_r1_r2(const string&, const Vector<string>&) const;
    string get_rv(const string&, const Vector<string>&) const;

    string replace_accented(const string&) const;

    Vector< Vector<string> > apply_stemmer(const Vector< Vector<string> >&) const;
    Vector< Vector<string> > apply_english_stemmer(const Vector< Vector<string> >&) const;
    Vector< Vector<string> > apply_spanish_stemmer(const Vector< Vector<string> >&) const;

    // Word bag

    /// It is a simplifying representation where a text(such as a sentence or a document) is represented
    /// as the bag(multiset) of its words, disregarding grammar and even word order but keeping multiplicity.

    size_t count(const Vector< Vector<string> >&) const;

    Vector<string> join(const Vector< Vector<string> >&) const;

    WordBag calculate_word_bag(const Vector< Vector<string> >&) const;

    WordBag calculate_word_bag_minimum_frequency(const Vector< Vector<string> >&, const size_t&) const;
    WordBag calculate_word_bag_minimum_percentage(const Vector< Vector<string> >&, const double&) const;
    WordBag calculate_word_bag_minimum_ratio(const Vector< Vector<string> >&, const double&) const;
    WordBag calculate_word_bag_total_frequency(const Vector< Vector<string> >&, const size_t&) const;
    WordBag calculate_word_bag_maximum_size(const Vector< Vector<string> >&, const size_t&) const;

    size_t calculate_weight(const Vector<string>&, const TextAnalytics::WordBag&) const;

    // Algorithms

    Vector< Vector<string> > preprocess(const Vector<string>&) const;

    Vector<size_t> get_words_number(const Vector< Vector<string> >&) const;

    Vector<size_t> get_sentences_number(const Vector<string>& documents) const;

    Vector<double> get_words_presence_percentage(const Vector< Vector<string> >&, const Vector<string>&) const;

    Matrix<string> calculate_combinated_words_frequency(const Vector< Vector<string> >&, const size_t&, const size_t&) const;

    Matrix<string> top_words_correlations(const Vector< Vector<string> >&, const double&, const Vector<size_t>&) const;

    Matrix<double> calculate_data_set(const Vector<string>&, const Vector<string>&, const TextAnalytics::WordBag&) const;

    // Binarize methods

    Vector<double> get_binary_vector(const Vector<string>&, const Vector<string>&) const;

    Matrix<double> get_binary_matrix(const Vector<string>&, const char& separator = ' ') const;

    Matrix<double> get_unique_binary_matrix(const Vector<string>&, const char&, const Vector<string>&) const;

private:

    void set_english_stop_words();
    void set_spanish_stop_words();
    void clear_stop_words();

    bool is_number(const string&) const;
    bool contains_number(const string&) const;

    bool is_email(const string&) const;

    bool starts_with(const string&, const string&) const;

    bool ends_with(const string&, const string&) const;
    bool ends_with(const string&, const Vector<string>&) const;

    // MEMBERS

    /// Documents language.

    Language lang = ENG;

    /// Words which are filtered out before or after processing of natural language data.

    Vector<string> stop_words;
};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2018 Artificial Intelligence Techniques, SL.
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

