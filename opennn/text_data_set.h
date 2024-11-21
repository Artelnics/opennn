//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E X T   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef TEXTDATASET_H
#define TEXTDATASET_H

#include "config.h"
#include "data_set.h"

namespace opennn
{

class TextDataSet : public DataSet
{

public:

    explicit TextDataSet();

    const Index& get_short_words_length() const;
    const Index& get_long_words_length() const;
    const Tensor<Index,1>& get_words_frequencies() const;

    Tensor<string, 2> get_text_data_file_preview() const;

    void set_short_words_length(const Index&);
    void set_long_words_length(const Index&);
    //    void set_words_frequencies(const Tensor<Index,1>&);

    void from_XML(const XMLDocument&);
    void to_XML(XMLPrinter&) const;

    Tensor<type, 1> sentence_to_data(const string&) const;

    void read_txt();

private:

    Index short_word_length = 2;

    Index long_word_length = 15;

    Tensor<Index, 1> word_frequencies;

    vector<string> stop_words  
    { "i", "me", "my", "myself", "we", "us", "our", "ours", "ourselves", "you", "u", "your", "yours", "yourself", "yourselves", "he",
     "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
     "what", "which", "who", "whom", "this", "that", "these", "those", "im", "am", "m", "is", "are", "was", "were", "be", "been", "being",
     "have", "has", "s", "ve", "re", "ll", "t", "had", "having", "do", "does", "did", "doing", "would", "d", "shall", "should", "could",
     "ought", "i'm", "you're", "he's", "she's", "it's", "we're", "they're", "i've", "you've", "we've", "they've", "i'd", "you'd", "he'd",
     "she'd", "we'd", "they'd", "i'll", "you'll", "he'll", "she'll", "we'll", "they'll", "isn't", "aren't", "wasn't", "weren't", "hasn't",
     "haven't", "hadn't", "doesn't", "don't", "didn't", "won't", "wouldn't", "shan't", "shouldn't", "can't", "cannot", "couldn't", "mustn't",
     "let's", "that's", "who's", "what's", "here's", "there's", "when's", "where's", "why's", "how's", "daren't", "needn't", "oughtn't",
     "mightn't", "shes", "its", "were", "theyre", "ive", "youve", "weve", "theyve", "id", "youd", "hed", "shed", "wed", "theyd",
     "ill", "youll", "hell", "shell", "well", "theyll", "isnt", "arent", "wasnt", "werent", "hasnt", "havent", "hadnt",
     "doesnt", "dont", "didnt", "wont", "wouldnt", "shant", "shouldnt", "cant", "cannot", "couldnt", "mustnt", "lets",
     "thats", "whos", "whats", "heres", "theres", "whens", "wheres", "whys", "hows", "darent", "neednt", "oughtnt",
     "mightnt", "a", "an", "the", "and", "n", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
     "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",
     "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",
     "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very" };
    
    Tensor<string, 2> text_data_file_preview;
};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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
