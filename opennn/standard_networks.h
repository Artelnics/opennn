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

/// @brief Factory neural network preconfigured for regression / function approximation.
class ApproximationNetwork : public NeuralNetwork
{

public:

    /// @brief Builds an approximation network with the given input, complexity and output shapes.
    ApproximationNetwork(const Shape& input_shape,
                         const Shape& complexity_dimensions,
                         const Shape& output_shape);
};

/// @brief Factory neural network preconfigured for tabular classification.
class ClassificationNetwork : public NeuralNetwork
{

public:

    /// @brief Builds a classification network with the given input, complexity and output shapes.
    ClassificationNetwork(const Shape& input_shape,
                          const Shape& complexity_dimensions,
                          const Shape& output_shape);
};

/// @brief Factory neural network preconfigured for time-series forecasting.
class ForecastingNetwork : public NeuralNetwork
{

public:

    /// @brief Builds a forecasting network with the given input, complexity and output shapes.
    ForecastingNetwork(const Shape& input_shape,
                       const Shape& complexity_dimensions,
                       const Shape& output_shape);
};

/// @brief Factory neural network preconfigured for auto-association (anomaly detection).
class AutoAssociationNetwork : public NeuralNetwork
{

public:

    /// @brief Builds an auto-association network with the given input, complexity and output shapes.
    AutoAssociationNetwork(const Shape& input_shape,
                           const Shape& complexity_dimensions,
                           const Shape& output_shape);
};

/// @brief Factory convolutional neural network preconfigured for image classification.
class ImageClassificationNetwork : public NeuralNetwork
{

public:

    /// @brief Builds an image classification network with the given input, complexity and output shapes.
    ImageClassificationNetwork(const Shape& input_shape,
                               const Shape& complexity_dimensions,
                               const Shape& output_shape);
};

/// @brief Factory residual neural network with a configurable number of blocks per stage.
class SimpleResNet : public NeuralNetwork
{

public:

    /// @brief Builds a residual network with the given input shape, per-stage block counts and output shape.
    /// @param input_shape Shape of the input tensor.
    /// @param blocks_per_stage Number of residual blocks at each stage.
    /// @param initial_filters Filter dimensions of the initial convolution.
    /// @param output_shape Shape of the output tensor.
    SimpleResNet(const Shape& input_shape,
                 const vector<Index>& blocks_per_stage,
                 const Shape& initial_filters,
                 const Shape& output_shape);
};

/// @brief Factory neural network reproducing the VGG-16 architecture.
class VGG16 final : public NeuralNetwork
{
public:

    /// @brief Builds a VGG-16 network with the given input and target shapes.
    VGG16(const Shape& input_shape, const Shape& target_shape);

    /// @brief Builds a VGG-16 network by loading it from the given file path.
    VGG16(const filesystem::path&);

    /// @brief Reconfigures the VGG-16 network with the given input and target shapes.
    void set(const Shape& input_shape, const Shape& target_shape);

};

/// @brief Factory neural network preconfigured for text classification.
class TextClassificationNetwork : public NeuralNetwork
{

public:

    /// @brief Builds a text classification network with the given input, complexity and output shapes.
    TextClassificationNetwork(const Shape& input_shape,
                              const Shape& complexity_dimensions,
                              const Shape& output_shape);
};

/// @brief Factory encoder-decoder Transformer neural network for sequence-to-sequence tasks.
class Transformer final : public NeuralNetwork
{
public:

    /// @brief Builds a Transformer with input/decoder sequence lengths, vocabularies, embedding, blocks and heads.
    Transformer(const Index = 0,
                Index = 0,
                Index = 0,
                Index = 0,
                Index = 0,
                Index = 0,
                Index = 0,
                Index = 0);

    /// @brief Reconfigures the Transformer with the supplied architecture parameters.
    void set(const Index = 0,
             Index = 0,
             Index = 0,
             Index = 0,
             Index = 0,
             Index = 0,
             Index = 0,
             Index = 0);

    /// @brief Returns the configured length of the encoder input sequence.
    Index get_input_sequence_length() const;

    /// @brief Returns the configured length of the decoder sequence.
    Index get_decoder_sequence_length() const;

    /// @brief Returns the embedding dimension used by the Transformer.
    Index get_embedding_dimension() const;

    /// @brief Returns the number of attention heads per block.
    Index get_heads_number() const;

    /// @brief Sets the dropout rate applied across the Transformer layers.
    void set_dropout_rate(const float);
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
