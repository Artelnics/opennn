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

class Transformer final : public NeuralNetwork
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

    Index get_input_sequence_length() const;
    Index get_decoder_sequence_length() const;
    Index get_embedding_dimension() const;
    Index get_heads_number() const;

    void set_dropout_rate(const type&);
    void set_input_vocabulary(const vector<string>&);
    void set_output_vocabulary(const vector<string>&);

    string calculate_outputs(const vector<string>&);

    Tensor<type, 3> calculate_outputs(const Tensor<type, 2>&, const Tensor<type, 2>&);

private:

    unordered_map<string, Index> input_vocabulary_map;
    unordered_map<Index, string> output_inverse_vocabulary_map;
};

};

#endif // TRANSFORMER_H
