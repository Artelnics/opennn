//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T A N D A R D   N E T W O R K S  C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file standard_networks.h
 * @brief Declares ready-made NeuralNetwork architectures for common
 *        learning tasks (regression, classification, forecasting, image
 *        classification, auto-association, ResNet, VGG-16, text
 *        classification, Transformer).
 *
 * Each class is a thin NeuralNetwork subclass whose constructor wires the
 * appropriate scaling, hidden, output and unscaling layers given a few
 * shape arguments. They serve as starting points; users can still mutate
 * the returned network with add_layer() / connect_layers() before
 * compile().
 */

#pragma once

#include "neural_network.h"

namespace opennn
{

/**
 * @class ApproximationNetwork
 * @brief Standard regression (function approximation) MLP.
 *
 * Builds Scaling -> stacked Dense layers (sized by @p complexity_dimensions)
 * -> Unscaling. Default activation is Tanh.
 */
class ApproximationNetwork : public NeuralNetwork
{

public:

    /**
     * @brief Constructs the network.
     * @param input_shape Per-sample input shape.
     * @param complexity_dimensions Number of neurons per hidden layer.
     * @param output_shape Per-sample output shape.
     */
    ApproximationNetwork(const Shape& input_shape,
                         const Shape& complexity_dimensions,
                         const Shape& output_shape);
};

/**
 * @class ClassificationNetwork
 * @brief Standard tabular classification MLP.
 *
 * Builds Scaling -> stacked Dense hidden layers -> Dense output layer
 * with Softmax (or Logistic for binary tasks). Inputs are scaled, outputs
 * are class probabilities and require no unscaling.
 */
class ClassificationNetwork : public NeuralNetwork
{

public:

    /**
     * @brief Constructs the network.
     * @param input_shape Per-sample input shape.
     * @param complexity_dimensions Number of neurons per hidden layer.
     * @param output_shape Per-sample output shape (number of classes).
     */
    ClassificationNetwork(const Shape& input_shape,
                          const Shape& complexity_dimensions,
                          const Shape& output_shape);
};

/**
 * @class ForecastingNetwork
 * @brief Standard time-series forecasting MLP.
 *
 * Builds Scaling -> stacked Recurrent / Dense layers -> Unscaling, sized
 * for inputs that already contain the desired past window.
 */
class ForecastingNetwork : public NeuralNetwork
{

public:

    /**
     * @brief Constructs the network.
     * @param input_shape Per-sample input shape (typically (past_steps, features)).
     * @param complexity_dimensions Number of neurons per hidden layer.
     * @param output_shape Per-sample output shape (typically (future_steps, targets)).
     */
    ForecastingNetwork(const Shape& input_shape,
                       const Shape& complexity_dimensions,
                       const Shape& output_shape);
};

/**
 * @class AutoAssociationNetwork
 * @brief Standard auto-encoder for outlier and novelty detection.
 *
 * Builds an encoder followed by a symmetric decoder; the network is
 * trained to reconstruct its inputs and the reconstruction error is used
 * as an outlier / novelty score.
 */
class AutoAssociationNetwork : public NeuralNetwork
{

public:

    /**
     * @brief Constructs the network.
     * @param input_shape Per-sample input shape.
     * @param complexity_dimensions Number of neurons per encoder hidden layer.
     * @param output_shape Per-sample output shape (typically equal to @p input_shape).
     */
    AutoAssociationNetwork(const Shape& input_shape,
                           const Shape& complexity_dimensions,
                           const Shape& output_shape);
};

/**
 * @class ImageClassificationNetwork
 * @brief Standard convolutional image classifier.
 *
 * Builds Scaling -> a stack of Convolutional + Pooling blocks (sized by
 * @p complexity_dimensions) -> Flatten -> Dense classifier head with Softmax.
 */
class ImageClassificationNetwork : public NeuralNetwork
{

public:

    /**
     * @brief Constructs the network.
     * @param input_shape Per-sample input shape (height, width, channels).
     * @param complexity_dimensions Filter counts for the convolutional stack.
     * @param output_shape Per-sample output shape (number of classes).
     */
    ImageClassificationNetwork(const Shape& input_shape,
                               const Shape& complexity_dimensions,
                               const Shape& output_shape);
};

/**
 * @class SimpleResNet
 * @brief Compact residual network for image classification.
 *
 * Builds an initial convolutional stem followed by a sequence of residual
 * blocks (Convolutional + Addition + Activation), grouped into stages
 * whose lengths are given by @p blocks_per_stage.
 */
class SimpleResNet : public NeuralNetwork
{

public:

    /**
     * @brief Constructs the network.
     * @param input_shape Per-sample input shape (height, width, channels).
     * @param blocks_per_stage Number of residual blocks per stage.
     * @param initial_filters Filter count and kernel shape of the initial conv stem.
     * @param output_shape Per-sample output shape (number of classes).
     */
    SimpleResNet(const Shape& input_shape,
                 const vector<Index>& blocks_per_stage,
                 const Shape& initial_filters,
                 const Shape& output_shape);
};

/**
 * @class VGG16
 * @brief VGG-16 architecture (Simonyan & Zisserman, 2014) for image classification.
 */
class VGG16 final : public NeuralNetwork
{
public:

    /**
     * @brief Constructs an untrained VGG-16.
     * @param input_shape Per-sample input shape (typically (224, 224, 3)).
     * @param target_shape Per-sample target shape (number of classes).
     */
    VGG16(const Shape& input_shape, const Shape& target_shape);

    /**
     * @brief Constructs a VGG-16 with parameters loaded from a saved model.
     * @param path Path to the JSON model file.
     */
    VGG16(const filesystem::path& path);

    /**
     * @brief Re-initializes the network with new shapes.
     * @param input_shape Per-sample input shape.
     * @param target_shape Per-sample target shape.
     */
    void set(const Shape& input_shape, const Shape& target_shape);

};

/**
 * @class TextClassificationNetwork
 * @brief Standard text classification model.
 *
 * Embedding -> sequence pooling -> Dense classifier head with Softmax.
 * Holds the input and output vocabularies needed to convert raw text into
 * token id batches before training and inference.
 */
class TextClassificationNetwork : public NeuralNetwork
{

public:

    /**
     * @brief Constructs the network.
     * @param input_shape Per-sample input shape (sequence_length,).
     * @param complexity_dimensions Per-layer sizing for the classifier head.
     * @param output_shape Per-sample output shape (number of classes).
     */
    TextClassificationNetwork(const Shape& input_shape,
                              const Shape& complexity_dimensions,
                              const Shape& output_shape);

private:

    /** @brief Input-side vocabulary used to encode raw text. */
    vector<string> input_vocabulary;
    /** @brief Output-side vocabulary (class labels). */
    vector<string> output_vocabulary;
};

/**
 * @class Transformer
 * @brief Encoder-decoder Transformer (Vaswani et al., 2017) for
 *        sequence-to-sequence modeling.
 */
class Transformer final : public NeuralNetwork
{
public:

    /**
     * @brief Constructs an untrained Transformer.
     * @param input_sequence_length Length of the encoder input sequence.
     * @param decoder_sequence_length Length of the decoder input sequence.
     * @param input_vocabulary_size Size of the input-side vocabulary.
     * @param output_vocabulary_size Size of the output-side vocabulary.
     * @param embedding_dimension Embedding (model) dimension.
     * @param heads_number Number of attention heads per layer.
     * @param feedforward_dimension Hidden size of the position-wise feed-forward sublayers.
     * @param layers_number Number of encoder (and decoder) layers.
     */
    Transformer(const Index input_sequence_length = 0,
                Index decoder_sequence_length = 0,
                Index input_vocabulary_size = 0,
                Index output_vocabulary_size = 0,
                Index embedding_dimension = 0,
                Index heads_number = 0,
                Index feedforward_dimension = 0,
                Index layers_number = 0);

    /**
     * @brief Re-initializes the Transformer with new dimensions.
     *
     * Arguments mirror the constructor.
     */
    void set(const Index input_sequence_length = 0,
             Index decoder_sequence_length = 0,
             Index input_vocabulary_size = 0,
             Index output_vocabulary_size = 0,
             Index embedding_dimension = 0,
             Index heads_number = 0,
             Index feedforward_dimension = 0,
             Index layers_number = 0);

    /** @brief Length of the encoder input sequence. */
    Index get_input_sequence_length() const;
    /** @brief Length of the decoder input sequence. */
    Index get_decoder_sequence_length() const;
    /** @brief Embedding (model) dimension. */
    Index get_embedding_dimension() const;
    /** @brief Number of attention heads per layer. */
    Index get_heads_number() const;

    /**
     * @brief Sets the dropout rate applied to the residual streams.
     *
     * Receives the dropout probability (0 disables dropout).
     */
    void set_dropout_rate(const float);
    /**
     * @brief Replaces the input-side vocabulary.
     *
     * Receives the new token list (order defines token ids).
     */
    void set_input_vocabulary(const vector<string>&);
    /**
     * @brief Replaces the output-side vocabulary.
     *
     * Receives the new token list (order defines token ids).
     */
    void set_output_vocabulary(const vector<string>&);

    /**
     * @brief Greedy decoding given a raw input string.
     * @param input Raw input text to encode.
     * @return Decoded output string.
     */
    string calculate_outputs(const string& input);

private:

    /** @brief Input-side vocabulary. */
    vector<string> input_vocabulary;
    /** @brief Output-side vocabulary. */
    vector<string> output_vocabulary;

    /** @brief Token-to-index lookup for the input vocabulary. */
    unordered_map<string, Index> input_vocabulary_map;
    /** @brief Index-to-token lookup for the output vocabulary. */
    unordered_map<Index, string> output_inverse_vocabulary_map;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
