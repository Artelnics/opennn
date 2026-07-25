//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C H A T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn_types.h"

namespace opennn
{

class NeuralNetwork;
class TextGenerationNetwork;
class TokenizerOperator;
class Transformer;
struct ForwardPropagation;

struct SamplingConfig
{
    float temperature = 1.0f;
    Index top_k = 0;
    float top_p = 1.0f;
    float repetition_penalty = 1.0f;
    Index maximum_tokens = 0;
};

Index sample_token(VectorR& probabilities,
                   const SamplingConfig&,
                   const vector<Index>& history);

enum class ReasoningMode
{
    Automatic,
    Enabled,
    Disabled
};

enum class GenerationChannel
{
    Reasoning,
    Content
};

enum class FinishReason
{
    Stop,
    MaximumTokens,
    ContextLimit
};

enum class ChatRole
{
    System,
    User,
    Assistant
};

struct ChatMessage
{
    ChatRole role = ChatRole::User;
    string content;
};

struct ChatOptions
{
    ReasoningMode reasoning_mode = ReasoningMode::Automatic;
    optional<SamplingConfig> sampling;
};

struct ChatDelta
{
    GenerationChannel channel = GenerationChannel::Content;
    string text;
};

using ChatCallback = function<void(const ChatDelta&)>;

struct ChatResponse
{
    string reasoning;
    string content;

    FinishReason finish_reason = FinishReason::MaximumTokens;

    Index prompt_tokens = 0;
    Index prefill_tokens = 0;
    Index generated_tokens = 0;
    Index reasoning_tokens = 0;
    Index content_tokens = 0;
    Index control_tokens = 0;

    double prefill_milliseconds = 0.0;
    double decode_milliseconds = 0.0;
};

struct GenerationParserSpec
{
    GenerationChannel initial_channel = GenerationChannel::Content;
    vector<Index> reasoning_start;
    vector<Index> reasoning_end;
    vector<vector<Index>> stop_sequences;
};

class ChatTemplate
{
public:
    virtual ~ChatTemplate() = default;

    virtual bool supports_reasoning() const noexcept = 0;
    virtual ReasoningMode default_reasoning_mode() const noexcept = 0;

    ReasoningMode resolve_reasoning_mode(ReasoningMode) const;

    virtual SamplingConfig default_sampling(ReasoningMode) const = 0;

    virtual vector<Index> render(const vector<ChatMessage>&,
                                 ReasoningMode,
                                 const TokenizerOperator&) const = 0;

    virtual GenerationParserSpec parser_spec(ReasoningMode,
                                              const TokenizerOperator&) const = 0;
};

class Qwen3ChatTemplate final : public ChatTemplate
{
public:
    bool supports_reasoning() const noexcept override { return true; }
    ReasoningMode default_reasoning_mode() const noexcept override
    {
        return ReasoningMode::Enabled;
    }

    SamplingConfig default_sampling(ReasoningMode) const override;

    vector<Index> render(const vector<ChatMessage>&,
                         ReasoningMode,
                         const TokenizerOperator&) const override;

    GenerationParserSpec parser_spec(ReasoningMode,
                                      const TokenizerOperator&) const override;
};

class GenerationParser
{
public:
    GenerationParser(const TokenizerOperator&,
                     const GenerationParserSpec&);

    bool push(Index, const ChatCallback& = {});
    void finish(const ChatCallback& = {});

    const string& get_reasoning() const noexcept { return reasoning_state.text; }
    const string& get_content() const noexcept { return content_state.text; }

    Index get_reasoning_tokens() const noexcept { return reasoning_tokens; }
    Index get_content_tokens() const noexcept { return content_tokens; }
    Index get_control_tokens() const noexcept { return control_tokens; }

private:
    struct ChannelState
    {
        vector<Index> ids;
        string text;
        string tail;
    };

    bool process_pending(const ChatCallback&, bool flush);
    void append_data_token(Index, const ChatCallback&);
    void emit_stable_delta(GenerationChannel, const ChatCallback&);
    ChannelState& channel_state(GenerationChannel) noexcept;

    const TokenizerOperator* tokenizer = nullptr;
    GenerationParserSpec spec;
    GenerationChannel channel = GenerationChannel::Content;
    bool incremental = false;
    vector<Index> pending;
    bool stopped = false;

    ChannelState reasoning_state;
    ChannelState content_state;
    Index reasoning_tokens = 0;
    Index content_tokens = 0;
    Index control_tokens = 0;
};

class ChatSession
{
public:

    explicit ChatSession(Transformer&);
    explicit ChatSession(TextGenerationNetwork&);

    ChatSession(NeuralNetwork&,
                const TokenizerOperator&,
                unique_ptr<ChatTemplate>,
                unsigned long long seed = 0);
    ~ChatSession();

    ChatSession(const ChatSession&) = delete;
    ChatSession& operator=(const ChatSession&) = delete;

    ChatResponse send(string_view user_message,
                      const ChatOptions& = {},
                      const ChatCallback& = {});

    void chat(const ChatOptions& = {});

    void set_messages(const vector<ChatMessage>&);
    const vector<ChatMessage>& get_messages() const noexcept;
    void clear();

    ReasoningMode resolve_reasoning_mode(ReasoningMode) const;
    SamplingConfig default_sampling(ReasoningMode) const;

    const ForwardPropagation& get_decode_propagation() const;

private:
    struct Impl;
    unique_ptr<Impl> impl;
};

}
