#include "tests/pch.h"

#include "opennn/network/chat.h"
#include "opennn/models/models.h"
#include "opennn/network/layers/tokenizer_layer.h"

#ifdef OPENNN_HAS_CUDA
#include "opennn/core/cuda/kernel_attention.cuh"
#include <bit>
#endif

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

TEST(SampleToken, LargeTopKAndTopPCompose)
{
    constexpr Index vocabulary_size = 65537;
    VectorR probabilities = VectorR::LinSpaced(vocabulary_size, 0.0f, 1.0f);

    SamplingConfig config;
    config.temperature = 1.0f;
    config.top_k = 1;
    config.top_p = 0.5f;

    EXPECT_EQ(sample_token(probabilities, config, {}), vocabulary_size - 1);
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

TEST(SampleToken, RepeatedCallsWithDifferentVocabularySizesAreIndependent)
{
    SamplingConfig config;
    config.top_k = 1;
    config.top_p = 0.5f;

    VectorR first(5);
    first << 0.1f, 0.2f, 0.3f, 0.4f, 0.9f;
    EXPECT_EQ(sample_token(first, config, {}), Index(4));

    VectorR second(2);
    second << 0.8f, 0.2f;
    EXPECT_EQ(sample_token(second, config, {}), Index(0));
}

#ifdef OPENNN_HAS_CUDA
TEST(SampleToken, CudaCandidatesMatchSortedLogitsAcrossBlocksAndTies)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";
    constexpr int vocabulary_size = 65539;
    constexpr int threads = 256;
    vector<float> logits(vocabulary_size);
    for (int i = 0; i < vocabulary_size; ++i) logits[size_t(i)] = float(i % 97) * 0.125f - 7.0f;
    logits[0] = 1000.0f; // Padding must never enter the candidates.
    for (int i : {17, 257, 32785, 65537}) logits[size_t(i)] = 20.0f;

    vector<vector<int>> sorted_blocks(LOGITS_SAMPLE_BLOCKS);
    for (int i = 1; i < vocabulary_size; ++i)
        sorted_blocks[size_t((i / threads) % LOGITS_SAMPLE_BLOCKS)].push_back(i);
    for (auto& indices : sorted_blocks)
        ranges::sort(indices, [&](int left, int right)
        {
            return logits[size_t(left)] == logits[size_t(right)] ? left < right
                : logits[size_t(left)] > logits[size_t(right)];
        });

    const auto stream = device::get_compute_stream();
    for (const Type precision : {Type::FP32, Type::BF16})
    {
        if (precision == Type::BF16 && device::cuda_compute_capability() < 80) continue;
        SCOPED_TRACE(precision == Type::FP32 ? "FP32" : "BF16");
        visit_type<Type::FP32, Type::BF16>(precision, [&]<typename T>()
        {
            vector<T> input(logits.begin(), logits.end());
            Buffer input_device(Device::CUDA), candidates_device(Device::CUDA);
            Buffer id_device(Device::CUDA), token_device(Device::CUDA);
            device::copy_async(input_device.ensure<T>(vocabulary_size), input.data(),
                Index(input.size() * sizeof(T)), device::CopyKind::HostToDevice, stream);
            float2* candidates = candidates_device.ensure<float2>(LOGITS_SAMPLE_BLOCKS * 32);
            int* id = id_device.ensure<int>(1);
            float* token = token_device.ensure<float>(1);

            for (const int k : {1, 7, 32})
            {
                SCOPED_TRACE(k);
                sample_logits_row_cuda<T>(vocabulary_size, k == 1 ? 0.0f : 1.0f, k, 1.0e-8f,
                    42, 3, input_device.as<T>(), candidates, id, token);
                vector<float2> actual(size_t(LOGITS_SAMPLE_BLOCKS * k));
                int sampled_id = -1;
                float sampled_token = -1.0f;
                device::copy_async(actual.data(), candidates, Index(actual.size() * sizeof(float2)),
                    device::CopyKind::DeviceToHost, stream);
                device::copy_async(&sampled_id, id, sizeof(int), device::CopyKind::DeviceToHost, stream);
                device::copy_async(&sampled_token, token, sizeof(float), device::CopyKind::DeviceToHost, stream);
                device::synchronize(stream);
                EXPECT_EQ(sampled_id, 17);
                EXPECT_FLOAT_EQ(sampled_token, 17.0f);
                for (int block = 0; block < LOGITS_SAMPLE_BLOCKS; ++block)
                    for (int rank = 0; rank < k; ++rank)
                    {
                        const int expected_id = sorted_blocks[size_t(block)][size_t(rank)];
                        const float2 candidate = actual[size_t(block * k + rank)];
                        EXPECT_EQ(std::bit_cast<int>(candidate.y), expected_id);
                        EXPECT_FLOAT_EQ(candidate.x, logits[size_t(expected_id)]);
                    }
            }
        });
    }
}
#endif

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
    EXPECT_EQ(make_tokenizer_operator("Qwen3")->get_kind(), "Qwen3");
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

TEST(TransformerInference, SequenceToSequenceSessionRequiresGpu)
{
    Transformer transformer(5, 4, 12, 14, 8, 2, 16, 1);

    transformer.set_input_vocabulary({"[PAD]", "[UNK]", "[START]", "[END]", "hello", "world"});
    transformer.set_target_vocabulary({"[PAD]", "[UNK]", "[START]", "[END]", "hola", "mundo"});

    EXPECT_FALSE(transformer.is_gpu());

    EXPECT_THROW(
        {
            ChatSession session(transformer);
        },
        runtime_error);
}

TEST(TransformerInference, DecoderOnlySessionRequiresGpu)
{
    TextGenerationNetwork network(6, 10, 8, 2, 16, 1, true);

    network.set_vocabulary({"[PAD]", "[UNK]", "alpha", "beta"});

    EXPECT_FALSE(network.is_gpu());

    EXPECT_THROW(
        {
            ChatSession session(network);
        },
        runtime_error);
}

TEST(TextClassificationNetworkTest, CalculatesOutputsFromDocuments)
{
    TextClassificationNetwork network(Shape{4, 3, 2}, Shape{1, 2}, Shape{2});

    Tensor<string, 1> documents(2);
    documents(0) = "alpha beta";
    documents(1) = "beta";

    EXPECT_THROW(network.calculate_text_outputs(documents), runtime_error);

    auto tokenizer = make_unique<WordLevelTokenizer>();
    tokenizer->set_vocabulary({"[PAD]", "[UNK]", "alpha", "beta"});
    network.set_tokenizer(std::move(tokenizer));

    const MatrixR outputs = network.calculate_text_outputs(documents);
    EXPECT_EQ(outputs.rows(), 2);
    EXPECT_EQ(outputs.cols(), 2);
    EXPECT_TRUE(outputs.allFinite());
}

TEST(TextClassificationNetworkTest, ClassifiesDocument)
{
    TextClassificationNetwork network(Shape{4, 3, 2}, Shape{1, 2}, Shape{2});

    auto tokenizer = make_unique<WordLevelTokenizer>();
    tokenizer->set_vocabulary({"[PAD]", "[UNK]", "alpha", "beta"});
    network.set_tokenizer(std::move(tokenizer));

    Variable emotion("emotion", "Target", VariableType::Categorical,
                     "None", {"sadness", "joy"});
    network.set_output_variables({emotion});
    network.set_parameters(VectorR::Zero(network.get_parameters_buffer_size()));

    const TextClassificationNetwork::Prediction prediction = network.classify("alpha beta");

    EXPECT_EQ(prediction.category, "sadness");
    EXPECT_FLOAT_EQ(prediction.confidence, 0.5f);
}

TEST(TextClassificationNetworkTest, ClassifiesBinaryDocument)
{
    TextClassificationNetwork network(Shape{4, 3, 2}, Shape{1, 2}, Shape{1});

    auto tokenizer = make_unique<WordLevelTokenizer>();
    tokenizer->set_vocabulary({"[PAD]", "[UNK]", "alpha", "beta"});
    network.set_tokenizer(std::move(tokenizer));

    Variable sentiment("sentiment", "Target", VariableType::Binary,
                       "None", {"negative", "positive"});
    network.set_output_variables({sentiment});
    network.set_parameters(VectorR::Zero(network.get_parameters_buffer_size()));

    Tensor<string, 1> documents(1);
    documents(0) = "alpha beta";
    const float positive_probability = network.calculate_text_outputs(documents)(0, 0);
    const TextClassificationNetwork::Prediction prediction = network.classify("alpha beta");

    EXPECT_EQ(prediction.category,
              positive_probability >= 0.5f ? "positive" : "negative");
    EXPECT_FLOAT_EQ(prediction.confidence,
                    max(positive_probability, 1.0f - positive_probability));
}
