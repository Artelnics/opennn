// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

#include "pch.h"

#include "opennn/chat.h"
#include "opennn/configuration.h"
#include "opennn/dense_layer.h"
#include "opennn/embedding_layer.h"
#include "opennn/forward_propagation.h"
#include "opennn/neural_network.h"
#include "opennn/random_utilities.h"
#include "opennn/standard_networks.h"
#include "opennn/tokenizer_operator.h"
#ifdef OPENNN_HAS_CUDA
#include "opennn/device_backend.h"
#endif

using namespace opennn;

namespace
{

class TemplateTokenizer final : public TokenizerOperator
{
public:
    TemplateTokenizer()
    {
        vector<string> tokens = {
            "[PAD]", "[UNK]",
            "<|im_start|>", "<|im_end|>", "<|endoftext|>",
            "<think>", "</think>",
            string(1, char(0xC3)), string(1, char(0xA9))
        };

        tokens.push_back("\n");
        for (int value = 32; value <= 126; ++value)
        {
            const string token(1, char(value));
            if (ranges::find(tokens, token) == tokens.end())
                tokens.push_back(token);
        }
        set_vocabulary(tokens);
    }

    vector<string> tokenize(string_view text) const override
    {
        static const vector<string> specials = {
            "<|endoftext|>", "<|im_start|>", "<|im_end|>",
            "</think>", "<think>"
        };

        vector<string> tokens;
        size_t position = 0;
        while (position < text.size())
        {
            const string* matched = nullptr;
            for (const string& special : specials)
                if (text.substr(position).starts_with(special))
                {
                    matched = &special;
                    break;
                }

            if (matched)
            {
                tokens.push_back(*matched);
                position += matched->size();
            }
            else
            {
                tokens.emplace_back(1, text[position]);
                ++position;
            }
        }
        return tokens;
    }

    string decode(const vector<Index>& ids) const override
    {
        string result;
        for (const Index id : ids)
            if (id > 0 && id < get_vocabulary_size())
                result += id_to_token(id);
        return result;
    }

    unique_ptr<TokenizerOperator> clone() const override
    {
        return make_unique<TemplateTokenizer>(*this);
    }

    string_view get_kind() const override { return "TemplateTest"; }

    Index id(string_view token) const
    {
        const auto iterator = get_vocabulary_map().find(string(token));
        if (iterator == get_vocabulary_map().end())
            throw runtime_error("Missing test token.");
        return iterator->second;
    }
};

class PlainChatTemplate final : public ChatTemplate
{
public:
    bool supports_reasoning() const noexcept override { return false; }
    ReasoningMode default_reasoning_mode() const noexcept override
    {
        return ReasoningMode::Disabled;
    }

    SamplingConfig default_sampling(ReasoningMode mode) const override
    {
        EXPECT_EQ(resolve_reasoning_mode(mode), ReasoningMode::Disabled);
        SamplingConfig config;
        config.temperature = 0.0f;
        config.maximum_tokens = 1;
        return config;
    }

    vector<Index> render(const vector<ChatMessage>& messages,
                         ReasoningMode mode,
                         const TokenizerOperator&) const override
    {
        EXPECT_EQ(resolve_reasoning_mode(mode), ReasoningMode::Disabled);
        return vector<Index>(messages.size() + 1, Index(2));
    }

    GenerationParserSpec parser_spec(
        ReasoningMode mode,
        const TokenizerOperator&) const override
    {
        EXPECT_EQ(resolve_reasoning_mode(mode), ReasoningMode::Disabled);
        return {};
    }
};

void make_tiny_decoder(NeuralNetwork& network,
                       Index sequence_length,
                       Index vocabulary_size)
{
    network.add_layer(
        make_unique<Embedding>(
            Shape{vocabulary_size, sequence_length}, 2, "embed_tokens"),
        {-1});
    network.add_layer(
        make_unique<opennn::Dense>(
            Shape{sequence_length, 2},
            Shape{vocabulary_size},
            "Identity",
            false,
            "lm_head"),
        {0});
    network.compile();
    network.set_parameters_random();
}

}

TEST(ChatTemplateTest, Qwen3GoldenPromptsAndModes)
{
    const TemplateTokenizer tokenizer;
    const Qwen3ChatTemplate chat_template;
    const vector<ChatMessage> messages = {
        {ChatRole::System, "Be brief."},
        {ChatRole::User, "Hello"},
        {ChatRole::Assistant, "Hi"},
        {ChatRole::User, "Why?"}
    };

    EXPECT_EQ(chat_template.resolve_reasoning_mode(ReasoningMode::Automatic),
              ReasoningMode::Enabled);
    EXPECT_EQ(chat_template.resolve_reasoning_mode(ReasoningMode::Enabled),
              ReasoningMode::Enabled);
    EXPECT_EQ(chat_template.resolve_reasoning_mode(ReasoningMode::Disabled),
              ReasoningMode::Disabled);

    const string common =
        "<|im_start|>system\nBe brief.<|im_end|>\n"
        "<|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\nHi<|im_end|>\n"
        "<|im_start|>user\nWhy?<|im_end|>\n"
        "<|im_start|>assistant\n";

    EXPECT_EQ(tokenizer.decode(chat_template.render(
                  messages, ReasoningMode::Enabled, tokenizer)),
              common + "<think>\n");
    EXPECT_EQ(tokenizer.decode(chat_template.render(
                  messages, ReasoningMode::Automatic, tokenizer)),
              common + "<think>\n");
    EXPECT_EQ(tokenizer.decode(chat_template.render(
                  messages, ReasoningMode::Disabled, tokenizer)),
              common + "<think>\n\n</think>\n\n");
}

TEST(ChatTemplateTest, Qwen3SamplingProfiles)
{
    const Qwen3ChatTemplate chat_template;

    const SamplingConfig thinking =
        chat_template.default_sampling(ReasoningMode::Automatic);
    EXPECT_FLOAT_EQ(thinking.temperature, 0.6f);
    EXPECT_EQ(thinking.top_k, 20);
    EXPECT_FLOAT_EQ(thinking.top_p, 0.95f);

    const SamplingConfig direct =
        chat_template.default_sampling(ReasoningMode::Disabled);
    EXPECT_FLOAT_EQ(direct.temperature, 0.7f);
    EXPECT_EQ(direct.top_k, 20);
    EXPECT_FLOAT_EQ(direct.top_p, 0.8f);
}

TEST(ChatTemplateTest, UnsupportedReasoningFailsBeforeGeneration)
{
    const PlainChatTemplate chat_template;

    EXPECT_EQ(chat_template.resolve_reasoning_mode(ReasoningMode::Automatic),
              ReasoningMode::Disabled);
    EXPECT_EQ(chat_template.resolve_reasoning_mode(ReasoningMode::Disabled),
              ReasoningMode::Disabled);
    EXPECT_THROW(
        chat_template.resolve_reasoning_mode(ReasoningMode::Enabled),
        runtime_error);
}

TEST(GenerationParserTest, SeparatesMultiTokenBoundariesAndStops)
{
    const TemplateTokenizer tokenizer;
    GenerationParserSpec spec;
    spec.initial_channel = GenerationChannel::Reasoning;
    spec.reasoning_start = {tokenizer.id("x"), tokenizer.id("y")};
    spec.reasoning_end = {tokenizer.id("z"), tokenizer.id("w")};
    spec.stop_sequences = {{
        tokenizer.id("!"), tokenizer.id("?")
    }};

    GenerationParser parser(tokenizer, spec);
    EXPECT_FALSE(parser.push(tokenizer.id("A")));
    EXPECT_FALSE(parser.push(tokenizer.id("z")));
    EXPECT_FALSE(parser.push(tokenizer.id("w")));
    EXPECT_FALSE(parser.push(tokenizer.id("B")));
    EXPECT_FALSE(parser.push(tokenizer.id("!")));
    EXPECT_TRUE(parser.push(tokenizer.id("?")));
    parser.finish();

    EXPECT_EQ(parser.get_reasoning(), "A");
    EXPECT_EQ(parser.get_content(), "B");
    EXPECT_EQ(parser.get_control_tokens(), 4);
}

TEST(GenerationParserTest, IncrementalChannelsMatchAccumulatedResponse)
{
    const TemplateTokenizer tokenizer;
    GenerationParserSpec spec;
    spec.initial_channel = GenerationChannel::Content;
    spec.reasoning_start = {tokenizer.id("x"), tokenizer.id("y")};
    spec.reasoning_end = {tokenizer.id("z"), tokenizer.id("w")};
    spec.stop_sequences = {{
        tokenizer.id("!"), tokenizer.id("?")
    }};

    vector<ChatDelta> deltas;
    const ChatCallback callback =
        [&deltas](const ChatDelta& delta) { deltas.push_back(delta); };
    GenerationParser parser(tokenizer, spec);

    const vector<Index> generated = {
        tokenizer.id("A"),
        tokenizer.id("x"), tokenizer.id("y"),
        tokenizer.id("B"),
        tokenizer.id("z"), tokenizer.id("w"),
        tokenizer.id("C"),
        // A repeated close in content is control, not user-visible text.
        tokenizer.id("z"), tokenizer.id("w"),
        tokenizer.id("!"), tokenizer.id("?")
    };
    for (const Index token : generated)
        if (parser.push(token, callback)) break;
    parser.finish(callback);

    EXPECT_EQ(parser.get_reasoning(), "B");
    EXPECT_EQ(parser.get_content(), "AC");
    EXPECT_EQ(parser.get_reasoning_tokens(), 1);
    EXPECT_EQ(parser.get_content_tokens(), 2);
    EXPECT_EQ(parser.get_control_tokens(), 8);

    string streamed_reasoning;
    string streamed_content;
    for (const ChatDelta& delta : deltas)
        (delta.channel == GenerationChannel::Reasoning
             ? streamed_reasoning
             : streamed_content) += delta.text;
    EXPECT_EQ(streamed_reasoning, parser.get_reasoning());
    EXPECT_EQ(streamed_content, parser.get_content());

    GenerationParser silent_parser(tokenizer, spec);
    for (const Index token : generated)
        if (silent_parser.push(token)) break;
    silent_parser.finish();
    EXPECT_EQ(silent_parser.get_reasoning(), parser.get_reasoning());
    EXPECT_EQ(silent_parser.get_content(), parser.get_content());
}

TEST(GenerationParserTest, StopBeforeReasoningCloseKeepsContentEmpty)
{
    const TemplateTokenizer tokenizer;
    GenerationParserSpec spec;
    spec.initial_channel = GenerationChannel::Reasoning;
    spec.reasoning_end = {tokenizer.id("z"), tokenizer.id("w")};
    spec.stop_sequences = {{
        tokenizer.id("!"), tokenizer.id("?")
    }};

    GenerationParser parser(tokenizer, spec);
    EXPECT_FALSE(parser.push(tokenizer.id("A")));
    EXPECT_FALSE(parser.push(tokenizer.id("!")));
    EXPECT_TRUE(parser.push(tokenizer.id("?")));
    parser.finish();

    EXPECT_EQ(parser.get_reasoning(), "A");
    EXPECT_TRUE(parser.get_content().empty());
}

TEST(GenerationParserTest, HoldsIncompleteUtf8AndMissingClose)
{
    const TemplateTokenizer tokenizer;
    GenerationParserSpec spec;
    spec.initial_channel = GenerationChannel::Reasoning;
    spec.reasoning_end = {tokenizer.id("z"), tokenizer.id("w")};

    vector<ChatDelta> deltas;
    const ChatCallback callback =
        [&deltas](const ChatDelta& delta) { deltas.push_back(delta); };
    GenerationParser parser(tokenizer, spec);

    EXPECT_FALSE(parser.push(tokenizer.id(
        string_view("\xC3", 1)), callback));
    EXPECT_TRUE(deltas.empty());
    EXPECT_FALSE(parser.push(tokenizer.id(
        string_view("\xA9", 1)), callback));
    ASSERT_EQ(deltas.size(), 1);
    EXPECT_EQ(deltas.front().channel, GenerationChannel::Reasoning);
    EXPECT_EQ(deltas.front().text, "é");

    parser.finish(callback);
    EXPECT_EQ(parser.get_reasoning(), "é");
    EXPECT_TRUE(parser.get_content().empty());
}

TEST(ChatSessionTest, StoresOnlyFinalContentAndTrimsTurns)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const TemplateTokenizer tokenizer;
    NeuralNetwork network;
    make_tiny_decoder(network, 8, tokenizer.get_vocabulary_size());
    ChatSession session(
        network, tokenizer, make_unique<PlainChatTemplate>(), 42);

    ChatOptions unsupported;
    unsupported.reasoning_mode = ReasoningMode::Enabled;
    EXPECT_THROW(session.send("not generated", unsupported),
                 runtime_error);
    EXPECT_TRUE(session.get_messages().empty());

    const ChatResponse first = session.send("first");
    EXPECT_TRUE(first.reasoning.empty());
    EXPECT_FALSE(first.content.empty());
    EXPECT_EQ(first.finish_reason, FinishReason::MaximumTokens);
    EXPECT_EQ(first.generated_tokens, 1);
    ASSERT_EQ(session.get_messages().size(), 2);
    EXPECT_EQ(session.get_messages()[1].role, ChatRole::Assistant);
    EXPECT_EQ(session.get_messages()[1].content, first.content);

    session.set_messages({
        {ChatRole::User, "u1"}, {ChatRole::Assistant, "a1"},
        {ChatRole::User, "u2"}, {ChatRole::Assistant, "a2"},
        {ChatRole::User, "u3"}, {ChatRole::Assistant, "a3"},
        {ChatRole::User, "u4"}, {ChatRole::Assistant, "a4"}
    });
    ChatOptions two_tokens;
    two_tokens.sampling =
        session.default_sampling(ReasoningMode::Disabled);
    two_tokens.sampling->maximum_tokens = 2;
    const ChatResponse trimmed =
        session.send("latest", two_tokens);
    EXPECT_EQ(trimmed.prompt_tokens, 8);
    EXPECT_EQ(trimmed.finish_reason, FinishReason::ContextLimit);
    ASSERT_EQ(session.get_messages().size(), 8);
    EXPECT_EQ(session.get_messages().front().content, "u2");

    Configuration::instance().set();
}

TEST(ChatSessionTest, PreservesSystemMessageWhenTrimming)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const TemplateTokenizer tokenizer;
    NeuralNetwork network;
    make_tiny_decoder(network, 8, tokenizer.get_vocabulary_size());
    ChatSession session(
        network, tokenizer, make_unique<PlainChatTemplate>(), 42);

    session.set_messages({
        {ChatRole::System, "system"},
        {ChatRole::User, "u1"}, {ChatRole::Assistant, "a1"},
        {ChatRole::User, "u2"}, {ChatRole::Assistant, "a2"},
        {ChatRole::User, "u3"}, {ChatRole::Assistant, "a3"},
        {ChatRole::User, "u4"}, {ChatRole::Assistant, "a4"}
    });

    const ChatResponse response = session.send("latest");
    EXPECT_LE(response.prompt_tokens, 8);
    ASSERT_FALSE(session.get_messages().empty());
    EXPECT_EQ(session.get_messages().front().role, ChatRole::System);
    EXPECT_EQ(session.get_messages().front().content, "system");
    EXPECT_EQ(session.get_messages()[1].content, "u3");

    Configuration::instance().set();
}

#ifdef OPENNN_HAS_CUDA
TEST(ChatSessionTest, ClassicDecoderUsesCommonResponseAndStreaming)
{
    if (!device::has_cuda_device()) GTEST_SKIP();

    Configuration::instance().set(Device::CUDA, Type::FP32);

    TextGenerationNetwork network(6, 10, 8, 2, 16, 1, true);
    network.set_vocabulary({
        "[PAD]", "[UNK]", "[START]", "[END]", "alpha",
        "beta", "gamma", "delta", "epsilon", "zeta"
    });

    ChatSession session(network);
    ChatOptions options;
    options.sampling =
        session.default_sampling(ReasoningMode::Disabled);
    options.sampling->maximum_tokens = 3;

    string streamed;
    const ChatResponse response = session.send(
        "alpha", options,
        [&](const ChatDelta& delta)
        {
            EXPECT_EQ(delta.channel, GenerationChannel::Content);
            streamed += delta.text;
        });

    EXPECT_EQ(response.reasoning_tokens, 0);
    EXPECT_EQ(response.generated_tokens, 3);
    EXPECT_EQ(response.finish_reason, FinishReason::MaximumTokens);
    EXPECT_EQ(streamed, response.content);

    ChatOptions unsupported;
    unsupported.reasoning_mode = ReasoningMode::Enabled;
    EXPECT_THROW(session.send("alpha", unsupported), runtime_error);

    Configuration::instance().set();
}

TEST(ChatSessionTest, SequenceToSequenceUsesCommonSendApi)
{
    if (!device::has_cuda_device()) GTEST_SKIP();

    Configuration::instance().set(Device::CUDA, Type::FP32);

    Transformer network(5, 4, 8, 8, 8, 2, 16, 1);
    network.set_input_vocabulary({
        "[PAD]", "[UNK]", "[START]", "[END]",
        "hello", "world", "small", "source"
    });
    network.set_target_vocabulary({
        "[PAD]", "[UNK]", "[START]", "[END]",
        "hola", "mundo", "small", "target"
    });

    ChatSession session(network);
    ChatOptions options;
    options.sampling =
        session.default_sampling(ReasoningMode::Disabled);
    options.sampling->maximum_tokens = 2;

    string streamed;
    const ChatResponse response = session.send(
        "hello world", options,
        [&](const ChatDelta& delta) { streamed += delta.text; });

    EXPECT_EQ(response.prompt_tokens, 4);
    EXPECT_LE(response.generated_tokens, 2);
    EXPECT_EQ(streamed, response.content);
    EXPECT_GE(response.prefill_milliseconds, 0.0);
    EXPECT_GE(response.decode_milliseconds, 0.0);

    Configuration::instance().set();
}

TEST(ChatSessionTest, ReusesCudaGraphAcrossFiveTurns)
{
    if (!device::has_cuda_device()) GTEST_SKIP();

    // The greedy 3-token turns must produce visible text, which depends on the
    // random weights: seed explicitly so the outcome does not depend on how
    // many global-RNG draws earlier tests consumed.
    set_seed(42);

    Configuration::instance().set(Device::CUDA, Type::BF16);

    const TemplateTokenizer tokenizer;
    Qwen3 network(
        64,
        tokenizer.get_vocabulary_size() - 1,
        32,
        1,
        4,
        2,
        8,
        64);
    network.set_parameters_random();
    network.upload_parameters_bf16_inference();

    ChatSession session(
        network, tokenizer, make_unique<PlainChatTemplate>(), 42);

    ChatOptions options;
    options.reasoning_mode = ReasoningMode::Disabled;
    options.sampling =
        session.default_sampling(ReasoningMode::Disabled);
    options.sampling->temperature = 0.0f;
    options.sampling->maximum_tokens = 3;

    void* graph_identity = nullptr;
    for (Index turn = 0; turn < 5; ++turn)
    {
        SCOPED_TRACE("turn " + to_string(turn));
        const ChatResponse response =
            session.send("turn " + to_string(turn), options);
        EXPECT_EQ(response.generated_tokens, 3);
        EXPECT_FALSE(response.content.empty());
        EXPECT_EQ(response.finish_reason,
                  FinishReason::MaximumTokens);

        const ForwardPropagation& decode =
            session.get_decode_propagation();
        ASSERT_TRUE(static_cast<bool>(decode.inference_graph_exec));
        if (!graph_identity)
            graph_identity = decode.inference_graph_exec.get();
        EXPECT_EQ(decode.inference_graph_exec.get(), graph_identity);
    }

    ASSERT_EQ(session.get_messages().size(), 10);
    for (size_t i = 1; i < session.get_messages().size(); i += 2)
        EXPECT_EQ(session.get_messages()[i].role,
                  ChatRole::Assistant);

    Configuration::instance().set();
}
#endif
