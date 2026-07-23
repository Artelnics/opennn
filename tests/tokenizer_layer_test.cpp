#include "pch.h"

#include "opennn/standard_networks.h"
#include "opennn/tokenizer_layer.h"
#include "opennn/text_dataset.h"

using namespace opennn;

TEST(SamplingConfig, Defaults)
{
    SamplingConfig config;

    EXPECT_FLOAT_EQ(config.temperature, 1.0f);
    EXPECT_EQ(config.top_k, Index(0));
    EXPECT_FLOAT_EQ(config.top_p, 1.0f);
    EXPECT_FLOAT_EQ(config.repetition_penalty, 1.0f);
    EXPECT_EQ(config.maximum_tokens, Index(0));
}

TEST(SamplingConfig, Assignment)
{
    SamplingConfig config;

    config.temperature = 0.5f;
    config.top_k = 40;
    config.top_p = 0.9f;
    config.repetition_penalty = 1.2f;
    config.maximum_tokens = 16;

    EXPECT_FLOAT_EQ(config.temperature, 0.5f);
    EXPECT_EQ(config.top_k, Index(40));
    EXPECT_FLOAT_EQ(config.top_p, 0.9f);
    EXPECT_FLOAT_EQ(config.repetition_penalty, 1.2f);
    EXPECT_EQ(config.maximum_tokens, Index(16));
}


TEST(SampleToken, GreedyReturnsArgmax)
{
    VectorR probabilities(4);
    probabilities << 0.1f, 0.2f, 0.6f, 0.1f;

    SamplingConfig config;
    config.temperature = 0.0f;

    EXPECT_EQ(sample_token(probabilities, config, {}), Index(2));
}


TEST(SampleToken, TopKOneKeepsOnlyArgmax)
{
    VectorR probabilities(4);
    probabilities << 0.1f, 0.5f, 0.3f, 0.1f;

    SamplingConfig config;
    config.temperature = 1.0f;
    config.top_k = 1;

    EXPECT_EQ(sample_token(probabilities, config, {}), Index(1));
}


TEST(SampleToken, TopPKeepsDominantToken)
{
    VectorR probabilities(3);
    probabilities << 0.7f, 0.2f, 0.1f;

    SamplingConfig config;
    config.temperature = 1.0f;
    config.top_p = 0.5f;

    EXPECT_EQ(sample_token(probabilities, config, {}), Index(0));
}


TEST(SampleToken, RepetitionPenaltyDemotesHistoryToken)
{
    VectorR probabilities(2);
    probabilities << 0.6f, 0.4f;

    SamplingConfig config;
    config.temperature = 1.0f;
    config.top_k = 1;
    config.repetition_penalty = 10.0f;

    EXPECT_EQ(sample_token(probabilities, config, {Index(0)}), Index(1));
}


TEST(SampleToken, DegenerateDistributionFallsBackToArgmax)
{
    VectorR probabilities(3);
    probabilities << 0.0f, 0.0f, 0.0f;

    SamplingConfig config;
    config.temperature = 1.0f;

    const Index sampled = sample_token(probabilities, config, {});

    EXPECT_GE(sampled, Index(0));
    EXPECT_LT(sampled, Index(3));
}


TEST(TokenizerLayer, IdentityPassthroughShape)
{
    Tokenizer tokenizer_layer(Shape{7});

    EXPECT_EQ(tokenizer_layer.get_output_shape(), Shape{7});
    EXPECT_EQ(tokenizer_layer.get_parameters_number(), Index(0));
    EXPECT_FALSE(tokenizer_layer.get_is_trainable());
    EXPECT_TRUE(tokenizer_layer.get_forward_specs(1).empty());
    EXPECT_EQ(tokenizer_layer.get_label(), "tokenizer");
}


TEST(TokenizerLayer, VocabularyRoundTrip)
{
    Tokenizer tokenizer_layer(Shape{4});

    EXPECT_TRUE(tokenizer_layer.get_vocabulary().empty());
    EXPECT_EQ(tokenizer_layer.get_vocabulary_size(), Index(0));

    const vector<string> vocabulary = {"[PAD]", "[UNK]", "[START]", "[END]", "alpha", "beta"};
    tokenizer_layer.set_vocabulary(vocabulary);

    EXPECT_EQ(tokenizer_layer.get_vocabulary(), vocabulary);
    EXPECT_EQ(tokenizer_layer.get_vocabulary_size(), Index(6));

    const auto& vocabulary_map = tokenizer_layer.get_vocabulary_map();
    const auto alpha_iterator = vocabulary_map.find("alpha");
    ASSERT_NE(alpha_iterator, vocabulary_map.end());
    EXPECT_EQ(alpha_iterator->second, Index(4));

    ASSERT_NE(tokenizer_layer.get_tokenizer(), nullptr);
    EXPECT_EQ(tokenizer_layer.get_tokenizer()->get_kind(), "WordLevel");
}


TEST(TokenizerLayer, SetTokenizerRegistersOperator)
{
    Tokenizer tokenizer_layer(Shape{4});

    EXPECT_TRUE(tokenizer_layer.get_operators().empty());

    tokenizer_layer.set_tokenizer(make_unique<WordLevelTokenizer>());

    ASSERT_EQ(tokenizer_layer.get_operators().size(), size_t(1));
    EXPECT_EQ(tokenizer_layer.get_operators()[0], tokenizer_layer.get_tokenizer());
}


TEST(TokenizerOperatorTest, BytePairCloneKeepsVocabularyAndMerges)
{
    BytePairTokenizer tokenizer;
    tokenizer.set_vocabulary({"[PAD]", "a", "b", "ab"});
    tokenizer.set_merges({"a b"});

    const unique_ptr<TokenizerOperator> cloned = tokenizer.clone();

    ASSERT_NE(cloned, nullptr);
    EXPECT_EQ(cloned->get_kind(), "BytePair");
    EXPECT_EQ(cloned->get_vocabulary(), tokenizer.get_vocabulary());

    auto* byte_pair = dynamic_cast<BytePairTokenizer*>(cloned.get());
    ASSERT_NE(byte_pair, nullptr);
    EXPECT_EQ(byte_pair->get_merges(), tokenizer.get_merges());

    EXPECT_EQ(cloned->encode("ab"), (vector<Index>{3}));
}


TEST(TokenizerOperatorTest, FactoryCreatesEachKind)
{
    EXPECT_EQ(make_tokenizer_operator("WordLevel")->get_kind(), "WordLevel");
    EXPECT_EQ(make_tokenizer_operator("WordPiece")->get_kind(), "WordPiece");
    EXPECT_EQ(make_tokenizer_operator("BytePair")->get_kind(), "BytePair");
    EXPECT_THROW(make_tokenizer_operator("Unknown"), runtime_error);
}


TEST(TransformerInference, DimensionGettersSurviveTokenizerLayers)
{
    const Index input_sequence_length = 5;
    const Index decoder_sequence_length = 4;
    const Index input_vocabulary_size = 12;
    const Index output_vocabulary_size = 14;
    const Index embedding_dimension = 8;
    const Index heads_number = 2;
    const Index feed_forward_dimension = 16;
    const Index layers_number = 1;

    Transformer transformer(input_sequence_length,
                            decoder_sequence_length,
                            input_vocabulary_size,
                            output_vocabulary_size,
                            embedding_dimension,
                            heads_number,
                            feed_forward_dimension,
                            layers_number);

    EXPECT_EQ(transformer.get_input_sequence_length(), input_sequence_length);
    EXPECT_EQ(transformer.get_decoder_sequence_length(), decoder_sequence_length);
    EXPECT_EQ(transformer.get_embedding_dimension(), embedding_dimension);
    EXPECT_EQ(transformer.get_heads_number(), heads_number);
    EXPECT_EQ(transformer.is_gpu(), false);
}


TEST(TransformerInference, ParametersNumberUnchangedByTokenizerLayers)
{
    Transformer transformer(6, 5, 30, 40, 8, 2, 16, 2);

    EXPECT_EQ(transformer.get_parameters_number(), Index(3928));

    TextGenerationNetwork generation_network(7, 50, 8, 2, 16, 2, false);

    EXPECT_EQ(generation_network.get_parameters_number(), Index(2050));
}


TEST(TransformerInference, NetworkVocabularySetters)
{
    Transformer transformer(5, 4, 12, 14, 8, 2, 16, 1);

    EXPECT_TRUE(transformer.get_input_vocabulary().empty());
    EXPECT_TRUE(transformer.get_target_vocabulary().empty());

    const vector<string> input_vocabulary = {"[PAD]", "[UNK]", "[START]", "[END]", "hello", "world"};
    const vector<string> target_vocabulary = {"[PAD]", "[UNK]", "[START]", "[END]", "hola", "mundo"};

    transformer.set_input_vocabulary(input_vocabulary);
    transformer.set_target_vocabulary(target_vocabulary);

    EXPECT_EQ(transformer.get_input_vocabulary(), input_vocabulary);
    EXPECT_EQ(transformer.get_target_vocabulary(), target_vocabulary);

    ASSERT_NE(transformer.get_input_tokenizer(), nullptr);
    EXPECT_EQ(transformer.get_input_tokenizer()->get_kind(), "WordLevel");
}


TEST(TransformerInference, DecodeRequiresGpu)
{
    Transformer transformer(5, 4, 12, 14, 8, 2, 16, 1);

    transformer.set_input_vocabulary({"[PAD]", "[UNK]", "[START]", "[END]", "hello", "world"});
    transformer.set_target_vocabulary({"[PAD]", "[UNK]", "[START]", "[END]", "hola", "mundo"});

    EXPECT_FALSE(transformer.is_gpu());

    EXPECT_THROW(transformer.decode("hello world"), runtime_error);
}


TEST(TransformerInference, GenerateRequiresGpu)
{
    TextGenerationNetwork network(6, 10, 8, 2, 16, 1, true);

    network.set_vocabulary({"[PAD]", "[UNK]", "alpha", "beta"});

    EXPECT_FALSE(network.is_gpu());

    EXPECT_THROW(network.generate("alpha"), runtime_error);
}


