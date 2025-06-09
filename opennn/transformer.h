//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A N S F O R M E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "neural_network.h"

namespace opennn
{

class Transformer : public NeuralNetwork
{
public:

    Transformer(const Index& decoder_length = 0,
                const Index& input_length = 0,
                const Index& decoder_dimensions = 0,
                const Index& input_dimension = 0,
                const Index& embedding_dimension = 0,
                const Index& perceptron_depth = 0,
                const Index& heads_number = 0,
                const Index& layers_number = 0);

    void set(const Index& decoder_length = 0,
             const Index& input_length = 0,
             const Index& decoder_dimensions = 0,
             const Index& input_dimension = 0,
             const Index& embedding_dimension = 0,
             const Index& perceptron_depth = 0,
             const Index& heads_number = 0,
             const Index& layers_number = 0);

    void set_dropout_rate(const type&);
    void set_input_vocabulary(const vector<string>&);
    void set_output_vocabulary(const vector<string>&);
    void set_input_length(const Index&);
    void set_decoder_length(const Index&);

    Index get_input_length() const;
    Index get_decoder_length() const;

    string calculate_outputs(const vector<string>&);
    string calculate_outputs(const string&, const bool&);

    Tensor<type, 3> calculate_outputs(const Tensor<type, 2>&, const Tensor<type, 2>&);

    void tokenize_whitespace(const vector<string>&, Tensor<type, 2>&);
    void tokenize_wordpiece(const vector<string>&, Tensor<type, 2>&);
    void detokenize_whitespace(Tensor<type, 2>&, ostringstream&);
    void detokenize_wordpiece(Tensor<type, 2>&, ostringstream&);

    vector<string> preprocess_language_document(const string& document, const bool& input)
    {
        vector<string> tokens;

        // if(!input)
        tokens.push_back("[START]");

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
        // if(!input)
       tokens.push_back("[END]");

        return tokens;
    }

private:

    Index input_length = 0;

    Index decoder_length = 0;

    type dropout_rate = type(0.1);

    vector<string> input_vocabulary;
    vector<string> output_vocabulary;
};

};

#endif // TRANSFORMER_H
