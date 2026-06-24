//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T A N D A R D   N E T W O R K S  C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "neural_network.h"
#include "pooling_layer.h"

namespace opennn
{

class ApproximationNetwork : public NeuralNetwork
{

public:

    ApproximationNetwork(const Shape& input_shape,
                         const Shape& complexity_dimensions,
                         const Shape& output_shape,
                         const string& hidden_activation = "Tanh");
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

class ForecastingLstmNetwork : public NeuralNetwork
{

public:

    ForecastingLstmNetwork(const Shape& input_shape,
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

#ifndef OPENNN_NO_VISION

class ImageClassificationNetwork : public NeuralNetwork
{

public:

    ImageClassificationNetwork(const Shape& input_shape,
                               const Shape& complexity_dimensions,
                               const Shape& output_shape);
};

class ResNet : public NeuralNetwork
{

public:

    ResNet(const Shape& input_shape,
           const vector<Index>& blocks_per_stage,
           const Shape& initial_filters,
           const Shape& output_shape,
           bool use_bottleneck = false);
};

class YoloNetwork : public NeuralNetwork
{
public:

    enum class Backbone { Vgg, DarknetTiny, DarknetTinyV3, Darknet53 };

    enum class ClassActivation { Softmax, Sigmoid };

    enum class HeadStyle { Single, FPN };

    // ReLU = the activation used since Phase 1 — preserves saved-weight effect.
    // LeakyReLU = Darknet/YOLO-v3 convention (slope 0.1) — applied uniformly
    // to every conv in the backbone (Vgg or DarknetTiny stages, residual
    // blocks, FPN lateral convs). The Detection layer's class activation and
    // the final logits stay as-is. Switching this changes forward outputs even
    // with identical parameters, so saved Phase 1/2 weights should not be
    // loaded across this flag.
    enum class BodyActivation { ReLU, LeakyReLU };

    YoloNetwork(const Shape& input_shape,
                Index classes_number,
                const vector<array<float, 2>>& anchors,
                Index grid_size = 13,
                Backbone backbone = Backbone::Vgg,
                ClassActivation class_activation = ClassActivation::Softmax,
                HeadStyle head_style = HeadStyle::Single,
                BodyActivation body_activation = BodyActivation::ReLU);
};

class TextClassificationNetwork : public NeuralNetwork
{

public:

    TextClassificationNetwork(const Shape& input_shape,
                              const Shape& complexity_dimensions,
                              const Shape& output_shape,
                              PoolingMethod pooling_method = PoolingMethod::AveragePooling);
};

class Transformer final : public NeuralNetwork
{
public:

    Transformer() = default;

    Transformer(Index input_sequence_length,
                Index decoder_sequence_length,
                Index input_vocabulary_size,
                Index output_vocabulary_size,
                Index embedding_dimension,
                Index heads_number,
                Index feed_forward_dimension,
                Index layers_number);

    Index get_input_sequence_length() const;
    Index get_decoder_sequence_length() const;
    Index get_embedding_dimension() const;
    Index get_heads_number() const;

    void set_dropout_rate(const float);
    void set_attention_sdpa_auto(bool);
    void set_attention_sdpa_min_sequence_length(Index);
};

#endif // OPENNN_NO_VISION

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
