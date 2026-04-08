//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T A N D A R D   N E T W O R K S  C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "neural_network.h"

namespace opennn
{

class ApproximationNetwork : public NeuralNetwork
{

public:

    ApproximationNetwork(const Shape& input_shape,
                         const Shape& complexity_dimensions,
                         const Shape& output_shape);
};

class ClassificationNetwork : public NeuralNetwork
{

public:

    ClassificationNetwork(const Shape& input_shape,
                          const Shape& complexity_dimensions,
                          const Shape& output_shape);
};

class ForecastingNetwork : public NeuralNetwork
{

public:

    ForecastingNetwork(const Shape& input_shape,
                       const Shape& complexity_dimensions,
                       const Shape& output_shape);
};

class AutoAssociationNetwork : public NeuralNetwork
{

public:

    AutoAssociationNetwork(const Shape& input_shape,
                           const Shape& complexity_dimensions,
                           const Shape& output_shape);
};

class ImageClassificationNetwork : public NeuralNetwork
{

public:

    ImageClassificationNetwork(const Shape& input_shape,
                               const Shape& complexity_dimensions,
                               const Shape& output_shape);
};

class SimpleResNet : public NeuralNetwork
{

public:

    SimpleResNet(const Shape& input_shape,
                 const vector<Index>& blocks_per_stage,
                 const Shape& initial_filters,
                 const Shape& output_shape);
};

class VGG16 final : public NeuralNetwork
{
public:

    VGG16(const Shape& input_shape, const Shape& target_shape);

    VGG16(const filesystem::path&);

    void set(const Shape& input_shape, const Shape& target_shape);

};

class TextClassificationNetwork : public NeuralNetwork
{

public:

    TextClassificationNetwork(const Shape& input_shape,
                              const Shape& complexity_dimensions,
                              const Shape& output_shape);

private:

    vector<string> input_vocabulary;
    vector<string> output_vocabulary;
};

class Transformer final : public NeuralNetwork
{
public:

    Transformer(const Index = 0,
                Index = 0,
                Index = 0,
                Index = 0,
                Index = 0,
                Index = 0,
                Index = 0,
                Index = 0);

    void set(const Index = 0,
             Index = 0,
             Index = 0,
             Index = 0,
             Index = 0,
             Index = 0,
             Index = 0,
             Index = 0);

    Index get_input_sequence_length() const;
    Index get_decoder_sequence_length() const;
    Index get_embedding_dimension() const;
    Index get_heads_number() const;

    void set_dropout_rate(const type);
    void set_input_vocabulary(const vector<string>&);
    void set_output_vocabulary(const vector<string>&);

    string calculate_outputs(const string&);

private:

    vector<string> input_vocabulary;
    vector<string> output_vocabulary;

    unordered_map<string, Index> input_vocabulary_map;
    unordered_map<Index, string> output_inverse_vocabulary_map;
};

} 

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
