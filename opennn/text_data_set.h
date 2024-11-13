//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E X T   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef TEXTDATASET_H
#define TEXTDATASET_H

#include <string>

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

    void from_XML(const tinyxml2::XMLDocument&);
    void to_XML(tinyxml2::XMLPrinter&) const;

    Tensor<type, 1> sentence_to_data(const string&) const;

    void read_txt();


private:

//    Separator text_separator = Separator::Tab;

    Index short_words_length = 2;

    Index long_words_length = 15;

    Tensor<Index, 1> words_frequencies;

    vector<string> stop_words;

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
