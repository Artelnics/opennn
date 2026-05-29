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

    // Vgg = the conv+pool stack used since Phase 1 (~6M params, no skip
    // connections). DarknetTiny = residual blocks à la Darknet-53 trimmed
    // for CPU tractability (~5.8M params). Default stays Vgg so existing call
    // sites and saved weight files are unaffected.
    enum class Backbone { Vgg, DarknetTiny };

    // Softmax = mutually-exclusive classes (YOLO v1/v2 + PASCAL VOC). Sigmoid
    // = independent per-class probabilities (YOLO v3+, multi-label datasets).
    // Default stays Softmax to preserve Phase 1/2 saved weights and behavior.
    enum class ClassActivation { Softmax, Sigmoid };

    // Single = one detection head at stride 32 (YOLO v1/v2 style, single grid).
    // FPN = three detection heads at strides 32 / 16 / 8 with top-down
    // upsample+concat skip path à la YOLO v3. FPN requires Backbone::DarknetTiny
    // (needs the stage-tap structure) and exactly 9 anchors (3 per scale,
    // assigned smallest→stride-8, largest→stride-32).
    enum class HeadStyle { Single, FPN };

    YoloNetwork(const Shape& input_shape,
                Index classes_number,
                const vector<array<float, 2>>& anchors,
                Index grid_size = 13,
                Backbone backbone = Backbone::Vgg,
                ClassActivation class_activation = ClassActivation::Softmax,
                HeadStyle head_style = HeadStyle::Single);
};

class TextClassificationNetwork : public NeuralNetwork
{

public:

    TextClassificationNetwork(const Shape& input_shape,
                              const Shape& complexity_dimensions,
                              const Shape& output_shape);
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
