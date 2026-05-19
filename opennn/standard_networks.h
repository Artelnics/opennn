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

// He et al. 2015 "Deep Residual Learning for Image Recognition" (ResNet-V1).
// Stem: 7x7 conv stride 2 (BN+ReLU) → 3x3 MaxPool stride 2.
// N stages, each with `blocks_per_stage[i]` residual blocks at
// `initial_filters[i]` filters. First block of each stage (except the first)
// downsamples by stride 2; the skip path projects with a 1x1 conv when shape
// changes, identity otherwise. BatchNorm fused inside every Convolutional.
//
// `use_bottleneck=false` → basic block (Conv3x3 → Conv3x3 → +skip → ReLU),
// suited for ResNet-18/34. `use_bottleneck=true` → bottleneck block
// (Conv1x1 → Conv3x3 → Conv1x1 with 4× channel expansion → +skip → ReLU),
// suited for ResNet-50/101/152. The post-addition ReLU is a standalone
// Activation layer, matching the textbook ResNet-V1 topology.
class ResNet : public NeuralNetwork
{

public:

    ResNet(const Shape& input_shape,
           const vector<Index>& blocks_per_stage,
           const Shape& initial_filters,
           const Shape& output_shape,
           bool use_bottleneck = false);
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

    void set_dropout_rate(const float);
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
