#include "pch.h"

#include "opennn/transformer_decoder.h"
#include "opennn/standard_networks.h"
#include "opennn/language_dataset.h"

using namespace opennn;

TEST(TransformerDecoder, SamplingConfigDefaults)
{
    TransformerDecoder::SamplingConfig config;

    EXPECT_FLOAT_EQ(config.temperature, 1.0f);
    EXPECT_EQ(config.top_k, Index(0));
    EXPECT_FLOAT_EQ(config.top_p, 1.0f);
    EXPECT_FLOAT_EQ(config.repetition_penalty, 1.0f);
    EXPECT_EQ(config.maximum_tokens, Index(0));
}


TEST(TransformerDecoder, SamplingConfigAssignment)
{
    TransformerDecoder::SamplingConfig config;

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

    TransformerDecoder::SamplingConfig greedy;
    greedy.temperature = 0.0f;

    EXPECT_FLOAT_EQ(greedy.temperature, 0.0f);
}


TEST(TransformerDecoder, TransformerDimensionGetters)
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


TEST(TransformerDecoder, ConstructorRequiresGpu)
{
    Transformer transformer(5, 4, 12, 14, 8, 2, 16, 1);

    LanguageDataset language_dataset;
    language_dataset.set_input_vocabulary({"[PAD]", "[UNK]", "[START]", "[END]", "hello", "world"});
    language_dataset.set_target_vocabulary({"[PAD]", "[UNK]", "[START]", "[END]", "hola", "mundo"});

    EXPECT_FALSE(transformer.is_gpu());

    EXPECT_THROW(TransformerDecoder(transformer, language_dataset), runtime_error);
}


TEST(TransformerDecoder, VocabularySetters)
{
    LanguageDataset language_dataset;

    EXPECT_TRUE(language_dataset.get_input_vocabulary_map().empty());
    EXPECT_TRUE(language_dataset.get_target_vocabulary().empty());

    const vector<string> input_vocabulary = {"[PAD]", "[UNK]", "[START]", "[END]", "alpha"};
    const vector<string> target_vocabulary = {"[PAD]", "[UNK]", "[START]", "[END]", "beta", "gamma"};

    language_dataset.set_input_vocabulary(input_vocabulary);
    language_dataset.set_target_vocabulary(target_vocabulary);

    EXPECT_EQ(language_dataset.get_input_vocabulary_size(), Index(input_vocabulary.size()));
    EXPECT_EQ(language_dataset.get_target_vocabulary_size(), Index(target_vocabulary.size()));

    EXPECT_FALSE(language_dataset.get_input_vocabulary_map().empty());
    EXPECT_EQ(language_dataset.get_target_vocabulary(), target_vocabulary);

    const auto& input_map = language_dataset.get_input_vocabulary_map();
    const auto start_iterator = input_map.find("[START]");
    ASSERT_NE(start_iterator, input_map.end());
    EXPECT_EQ(start_iterator->second, Index(2));
}
