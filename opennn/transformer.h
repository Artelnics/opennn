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

    Transformer(const Index& input_length = 0,
                const Index& decoder_length = 0,
                const Index& input_dimensions = 0,
                const Index& context_dimension = 0,
                const Index& embedding_dimension = 0,
                const Index& perceptron_depth = 0,
                const Index& heads_number = 0,
                const Index& layers_number = 0);

    void set(const Index& input_length = 0,
             const Index& decoder_length = 0,
             const Index& input_dimensions = 0,
             const Index& context_dimension = 0,
             const Index& embedding_dimension = 0,
             const Index& perceptron_depth = 0,
             const Index& heads_number = 0,
             const Index& layers_number = 0);

    void set_dropout_rate(const type&);
    void set_input_vocabulary(const unordered_map<string, Index>&);
    void set_output_vocabulary(const unordered_map<string, Index>&);

    string calculate_outputs(const vector<string>&);

    Tensor<type, 3> calculate_outputs(const Tensor<type, 2>&, const Tensor<type, 2>&);

    void tokenize_wordpiece(const vector<string>&, Tensor<type, 2>&);
    void detokenize_wordpiece(Tensor<type, 2>&, ostringstream&);

private:

    Index input_length = 0;

    Index decoder_length = 0;

    type dropout_rate = 0;

    unordered_map<string, Index> input_vocabulary;
    unordered_map<string, Index> output_vocabulary;
};

};

#endif // TRANSFORMER_H
