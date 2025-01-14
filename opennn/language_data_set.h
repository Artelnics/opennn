//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A N G U A G E   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef LANGUAGEDATASET_H
#define LANGUAGEDATASET_H

#include "data_set.h"

namespace opennn
{

class LanguageDataSet : public DataSet
{

public:

    explicit LanguageDataSet(const dimensions& = {0}, const dimensions& = {0});

    explicit LanguageDataSet(const filesystem::path&);

    const unordered_map<string, Index>& get_input_vocabulary() const;
    const unordered_map<string, Index>& get_target_vocabulary() const;

    Index get_input_vocabulary_size() const;
    Index get_target_vocabulary_size() const;

    void set_input_vocabulary(const unordered_map<string, Index>&);
    void set_target_vocabulary(const unordered_map<string, Index>&);

    void set_data_random() override;

    void read_csv() override;

    Index count_non_empty_lines() const;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

    vector<string> tokenize(const string& document)
    {
        vector<string> tokens;

        tokens.push_back("[END]");

        string currentToken;

        for (char c : document)
        {
            if (isalnum(c))
            {
                // Add alphanumeric characters to the current token
                currentToken += tolower(c);
            }
            else
            {
                // If the current token is not empty, add it to the tokens list
                if (!currentToken.empty())
                {
                    tokens.push_back(currentToken);
                    currentToken.clear();
                }
                // Treat punctuation as a separate token

                if (ispunct(c))
                {
                    tokens.push_back(string(1, c));
                }
                else if (isspace(c))
                {
                    // Ignore spaces, they just delimit tokens
                }
            }
        }

        // Add the last token if it's not empty
        if (!currentToken.empty())
            tokens.push_back(currentToken);

        // Add [END] token
        tokens.push_back("[END]");

        return tokens;
    }


    unordered_map<string, Index> create_vocabulary(const vector<vector<string>>& document_tokens)
    {
        unordered_map<string, Index> vocabulary;
        Index id = 0;

        vocabulary["[PAD]"] = id++;
        vocabulary["[UNK]"] = id++;
        vocabulary["[START]"] = id++;
        vocabulary["[END]"] = id++;

        for (const auto& document : document_tokens)
            for (const auto& token : document)
                if (vocabulary.find(token) == vocabulary.end())
                    vocabulary[token] = id++;

        return vocabulary;
    }


    void print_vocabulary(const unordered_map<std::string, Index>& vocabulary)
    {
        for (const auto& entry : vocabulary)
            cout << entry.first << " : " << entry.second << "\n";
    }

private:

    unordered_map<string, Index> input_vocabulary;
    unordered_map<string, Index> target_vocabulary;

    Index maximum_input_length = 0;

    Index maximum_target_length = 0;

    const Index target_vocabulary_size = 8000;
    const vector<string> reserved_tokens = { "[PAD]", "[UNK]", "[START]", "[END]" };

    struct WordpieceAlgorithmParameters
    {
        Index upper_threshold = 0;
        Index lower_threshold = 0;
        Index iterations_number = 0;
        Index max_input_tokens = 0;
        Index max_token_length = 0;
        Index max_unique_characters = 0;
        Index vocabulary_size = 0;
        float slack_ratio = 0;
        bool include_joiner_token = false;
        string joiner;
        vector<string> reserved_tokens;
    };
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
