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
//#include "forward_propagation.h"

namespace opennn
{

//struct TransformerForwardPropagation;
//struct TransformerBackPropagation;

class Transformer : public NeuralNetwork
{
public:

    // Constructors


    // explicit Transformer(const Tensor<Index, 1>& = Tensor<Index, 1>());

    Transformer();

    Transformer(const Tensor<Index, 1>&);

    Transformer(const initializer_list<Index>&);

    explicit Transformer(const dimensions&, const dimensions&, const vector <Index>&);

    // Index get_input_length() const
    // {
    //     return 0;
    // }

    void set(const Tensor<Index, 1>&);

    void set(const initializer_list<Index>&);

    void set(const Index& input_length,
             const Index& context_length,
             const Index& input_dimensions,
             const Index& context_dimension,
             const Index& embedding_depth,
             const Index& perceptron_depth,
             const Index& heads_number,
             const Index& layers_number);

    void set_dropout_rate(const type&);
    void set_input_vocabulary(const vector<string>&);
    void set_context_vocabulary(const vector<string>&);

    string calculate_outputs(const vector<string>&);

    Tensor<type, 3> calculate_outputs(const Tensor<type, 2>&, const Tensor<type, 2>&);

//    void tokenize_whitespace(const vector<string>&, Tensor<type, 2>&);
    void tokenize_wordpiece(const vector<string>&, Tensor<type, 2>&);

    void detokenize_whitespace(Tensor<type, 2>&, ostringstream&);
    void detokenize_wordpiece(Tensor<type, 2>&, ostringstream&);

    void load_transformer(const string&);

private:

    string name = "transformer";

    Index input_length = 0;

    Index context_length = 0;

    Index input_dimensions_xxx = 0;

    Index context_dimension_xxx = 0;

    Index embedding_depth = 0;

    Index perceptron_depth = 0;

    Index heads_number = 0;

    Index layers_number = 0;

    type dropout_rate = 0;

    vector<string> input_vocabulary;
    vector<string> context_vocabulary;
};


// struct TransformerForwardPropagation : ForwardPropagation
// {
//     // Constructors

//     TransformerForwardPropagation() {}

//     TransformerForwardPropagation(const Index& new_batch_samples, NeuralNetwork* new_neural_network)
//     {
//         set(new_batch_samples, new_neural_network);
//     }

//     void set(const Index& new_batch_samples, NeuralNetwork* new_neural_network);

//     void print() const;

//     Index batch_samples_number = 0;

//     Tensor<unique_ptr<LayerForwardPropagation>, 1> layers;
// };
};

#endif // TRANSFORMER_H
