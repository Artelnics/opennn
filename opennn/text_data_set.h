//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E X T   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef TEXTDATASET_H
#define TEXTDATASET_H

// System includes

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <limits>
#include <math.h>

// OpenNN includes

#include "config.h"
#include "data_set.h"

namespace opennn
{

class TextDataSet : public DataSet
{

public:

    // DEFAULT CONSTRUCTOR

    explicit TextDataSet();

    string get_text_separator_string() const;

    const Index& get_short_words_length() const;
    const Index& get_long_words_length() const;
    const Tensor<Index,1>& get_words_frequencies() const;

    void set_text_separator(const Separator&);
    void set_text_separator(const string&);

    Tensor<string, 2> get_text_data_file_preview() const;

    void set_short_words_length(const Index&);
    void set_long_words_length(const Index&);
    void set_words_frequencies(const Tensor<Index,1>&);

    void from_XML(const tinyxml2::XMLDocument&);
    void write_XML(tinyxml2::XMLPrinter&) const;

    Tensor<type,1> sentence_to_data(const string&) const;

    void read_txt();

    /// Sets the words that will be removed from the documents.

    void set_english_stop_words()
    {
        stop_words.resize(242);

        stop_words.setValues(
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
        "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very" });
    }



    void set_spanish_stop_words()
    {
        stop_words.resize(327);

        stop_words.setValues(
        { "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para", "con", "no", "una", "su", "al",
        "es", "lo", "como", "m�s", "mas", "pero", "sus", "le", "ya", "o", "fue", "este", "ha", "si", "s�", "porque", "esta", "son",
        "entre", "est�", "cuando", "muy", "a�n", "aunque", "sin", "sobre", "ser", "tiene", "tambi�n", "me", "hasta", "hay", "donde", "han", "quien",
        "est�n", "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra", "otros", "fueron", "ese", "eso", "hab�a",
        "ante", "ellos", "e", "esto", "m�", "antes", "algunos", "qu�", "unos", "yo", "otro", "otras", "otra", "�l", "tanto", "esa",
        "estos", "mucho", "quienes", "nada", "muchos", "cual", "sea", "poco", "ella", "estar", "haber", "estas", "estaba", "estamos",
        "algunas", "algo", "nosotros", "mi", "mis", "t�", "te", "ti", "tu", "tus", "ellas", "nosotras", "vosotros", "vosotras", "os",
        "m�o", "m�a", "m�os", "m�as", "tuyo", "tuya", "tuyos", "tuyas", "suyo", "suya", "suyos", "suyas", "nuestro", "nuestra", "nuestros",
        "nuestras", "vuestro", "vuestra", "vuestros", "vuestras", "esos", "esas", "estoy", "est�s", "est�", "estamos", "est�is", "est�n",
        "est�", "est�s", "estemos", "est�is", "est�n", "estar�", "estar�s", "estar�", "estaremos", "estar�is", "estar�n", "estar�a",
        "estar�as", "estar�amos", "estar�ais", "estar�an", "estaba", "estabas", "est�bamos", "estabais", "estaban", "estuve", "estuviste",
        "estuvo", "estuvimos", "estuvisteis", "estuvieron", "estuviera", "estuvieras", "estuvi�ramos", "estuvierais", "estuvieran", "estuviese",
        "estuvieses", "estuvi�semos", "estuvieseis", "estuviesen", "estando", "estado", "estada", "estados", "estadas", "estad", "he",
        "has", "ha", "hemos", "hab�is", "han", "haya", "hayas", "hayamos", "hay�is", "hayan", "habr�", "habr�s", "habr�", "habremos",
        "habr�is", "habr�n", "habr�a", "habr�as", "habr�amos", "habr�ais", "habr�an", "hab�a", "hab�as", "hab�amos", "hab�ais", "hab�an",
        "hube", "hubiste", "hubo", "hubimos", "hubisteis", "hubieron", "hubiera", "hubieras", "hubi�ramos", "hubierais", "hubieran",
        "hubiese", "hubieses", "hubi�semos", "hubieseis", "hubiesen", "habiendo", "habido", "habida", "habidos", "habidas", "soy", "eres",
        "es", "somos", "sois", "son", "sea", "seas", "seamos", "se�is", "sean", "ser�", "ser�s", "ser�", "seremos", "ser�is", "ser�n",
        "ser�a", "ser�as", "ser�amos", "ser�ais", "ser�an", "era", "eras", "�ramos", "erais", "eran", "fui", "fuiste", "fue", "fuimos",
        "fuisteis", "fueron", "fuera", "fueras", "fu�ramos", "fuerais", "fueran", "fuese", "fueses", "fu�semos", "fueseis", "fuesen", "siendo",
        "sido", "tengo", "tienes", "tiene", "tenemos", "ten�is", "tienen", "tenga", "tengas", "tengamos", "teng�is", "tengan", "tendr�",
        "tendr�s", "tendr�", "tendremos", "tendr�is", "tendr�n", "tendr�a", "tendr�as", "tendr�amos", "tendr�ais", "tendr�an", "ten�a",
        "ten�as", "ten�amos", "ten�ais", "ten�an", "tuve", "tuviste", "tuvo", "tuvimos", "tuvisteis", "tuvieron", "tuviera", "tuvieras",
        "tuvi�ramos", "tuvierais", "tuvieran", "tuviese", "tuvieses", "tuvi�semos", "tuvieseis", "tuviesen", "teniendo", "tenido", "tenida",
        "tenidos", "tenidas", "tened" });
    }


private:

    Separator text_separator = Separator::Tab;

    Index short_words_length = 2;

    Index long_words_length = 15;

    Tensor<Index, 1> words_frequencies;

    Tensor<string, 1> stop_words;

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
