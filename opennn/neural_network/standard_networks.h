//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T A N D A R D   N E T W O R K S  C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/neural_network/neural_network.h"
#include "opennn/neural_network/layers/pooling_layer.h"
#include "opennn/neural_network/detection_head.h"

namespace opennn
{

class TokenizerOperator;

class ApproximationNetwork : public NeuralNetwork
{

public:

    ApproximationNetwork(const Shape&,
                         const Shape&,
                         const Shape&,
                         const string& hidden_activation = "Tanh");
};

class ClassificationNetwork : public NeuralNetwork
{

public:

    ClassificationNetwork(const Shape&,
                          const Shape&,
                          const Shape&,
                          const string& hidden_activation = "Tanh");
};

class ForecastingNetwork : public NeuralNetwork
{

public:

    ForecastingNetwork(const Shape&,
                       const Shape&,
                       const Shape&);
};

class ForecastingLstmNetwork : public NeuralNetwork
{

public:

    ForecastingLstmNetwork(const Shape&,
                           const Shape&,
                           const Shape&);
};

class AutoAssociationNetwork : public NeuralNetwork
{

public:

    AutoAssociationNetwork(const Shape&,
                           const Shape&,
                           const Shape&);

    AutoAssociationNetwork(const Shape&,
                           const Shape&,
                           const string&,
                           const string&);
};

#ifndef OPENNN_NO_VISION

class ImageClassificationNetwork : public NeuralNetwork
{

public:

    ImageClassificationNetwork(const Shape&,
                               const Shape&,
                               const Shape&);
};

class ResNet : public NeuralNetwork
{

public:

    ResNet(const Shape&,
           const vector<Index>&,
           const Shape&,
           const Shape&,
           bool use_bottleneck = false);
};

class YoloNetwork : public NeuralNetwork
{
public:

    enum class Backbone { Vgg, DarknetTiny, DarknetTinyV3, Darknet53, CSPDarknet53, CSPDarknet53v11 };

    using ClassActivation = DetectionClassActivation;

    enum class HeadStyle { Single, FPN, PANet, FPNv8 };

    enum class BodyActivation { ReLU, LeakyReLU, SiLU };

    enum class ModelSize { n, s, m, l, x };

    YoloNetwork(const Shape&,
                Index,
                const vector<array<float, 2>>&,
                Index grid_size = 13,
                Backbone backbone = Backbone::Vgg,
                ClassActivation class_activation = ClassActivation::Softmax,
                HeadStyle head_style = HeadStyle::Single,
                BodyActivation body_activation = BodyActivation::ReLU,
                bool use_sppf = false,
                Index reg_max = 1,
                ModelSize model_size = ModelSize::l);
};

class TextClassificationNetwork : public NeuralNetwork
{

public:

    TextClassificationNetwork(const Shape&,
                              const Shape&,
                              const Shape&,
                              PoolingMethod pooling_method = PoolingMethod::AveragePooling);

    MatrixR calculate_text_outputs(const Tensor<string, 1>&);

    void set_tokenizer(unique_ptr<TokenizerOperator>);
    const TokenizerOperator* get_tokenizer() const;
};

class Transformer final : public NeuralNetwork
{
public:

    Transformer();

    Transformer(Index,
                Index,
                Index,
                Index,
                Index,
                Index,
                Index,
                Index);

    explicit Transformer(const filesystem::path&);

    Index get_input_sequence_length() const { return get_layer("encoder_embedding")->get_input_shape()[0]; }
    Index get_decoder_sequence_length() const { return get_layer("decoder_embedding")->get_input_shape()[0]; }

    void set_dropout_rate(const float);
    void set_attention_sdpa_min_sequence_length(Index);

    void set_input_vocabulary(const vector<string>&);
    void set_target_vocabulary(const vector<string>&);
    const TokenizerOperator* get_input_tokenizer() const;
    const TokenizerOperator* get_target_tokenizer() const;
    const vector<string>& get_input_vocabulary() const;
    const vector<string>& get_target_vocabulary() const;

};

class TextGenerationNetwork final : public NeuralNetwork
{
public:

    TextGenerationNetwork();

    TextGenerationNetwork(Index,
                          Index,
                          Index,
                          Index,
                          Index,
                          Index,
                          bool pre_normalization = false,
                          bool scale_embedding = true,
                          bool learned_positional = false,
                          const string& feed_forward_activation = "ReLU");

    explicit TextGenerationNetwork(const filesystem::path&);

    Index get_sequence_length() const { return get_layer("embedding")->get_input_shape()[0]; }

    void set_dropout_rate(const float);
    void set_attention_sdpa_auto(bool);

    void set_tokenizer(unique_ptr<TokenizerOperator>);
    void set_vocabulary(const vector<string>&);
    const TokenizerOperator* get_tokenizer() const;

};

class Qwen3 final : public NeuralNetwork
{
public:

    Qwen3();

    Qwen3(Index sequence_length,
          Index vocabulary_size,
          Index hidden_size,
          Index layers_number,
          Index query_heads,
          Index key_value_heads,
          Index head_dimension,
          Index intermediate_size,
          float rope_theta = 1000000.0f,
          float rms_epsilon = 1.0e-6f);
};

class Bert final : public NeuralNetwork
{
public:

    Bert();

    Bert(Index sequence_length,
         Index vocabulary_size,
         Index hidden_size,
         Index heads_number,
         Index intermediate_size,
         Index layers_number,
         Index type_vocabulary_size = 2);
};

class BertForSequenceClassification final : public NeuralNetwork
{
public:

    BertForSequenceClassification();

    BertForSequenceClassification(Index sequence_length,
                                  Index vocabulary_size,
                                  Index hidden_size,
                                  Index heads_number,
                                  Index intermediate_size,
                                  Index layers_number,
                                  Index labels_number,
                                  Index type_vocabulary_size = 2);

    void set_dropout_rate(const float);
};

#endif

Index load_darknet_backbone(NeuralNetwork&, const filesystem::path&, Index);
Index load_darknet_backbone_v11(NeuralNetwork&, const filesystem::path&);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
