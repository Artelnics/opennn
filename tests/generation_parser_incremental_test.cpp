#include "opennn/chat.h"
#include "opennn/tokenizer_operator.h"

#include <gtest/gtest.h>

using namespace opennn;
using namespace std;

// The incremental decode path in GenerationParser engages only for tokenizers
// with supports_incremental_decode() == true (BytePairTokenizer). The pinned
// GenerationParserTest cases use a test tokenizer that takes the full-redecode
// fallback, so these tests cover the incremental path against the fallback's
// contract: emitted deltas concatenate to exactly decode() of the data ids.

namespace
{

// [PAD]=0, he=1, llo=2, " wor"=3, ld=4, C3=5, A9=6, <s>=7, </s>=8, !=9, ?=10
// Byte-level BPE vocab entries are byte-encoded: each raw byte b is stored as
// the UTF-8 encoding of the codepoint byte_encoder[b]. Bytes 0xC3 and 0xA9 map
// to themselves, so their entries are the UTF-8 of U+00C3 and U+00A9. Decoding
// ids {5, 6} therefore yields the raw bytes C3 A9 — one "é" split across two
// tokens, which is what the withholding test needs.
BytePairTokenizer make_tokenizer()
{
    BytePairTokenizer tokenizer;
    tokenizer.set_vocabulary({"[PAD]", "he", "llo", " wor", "ld",
                              "\xC3\x83", "\xC2\xA9", "<s>", "</s>", "!", "?"});
    return tokenizer;
}

}

TEST(GenerationParserIncremental, BpeDecodeEqualsConcatOfDecodeToken)
{
    const BytePairTokenizer tokenizer = make_tokenizer();
    ASSERT_TRUE(tokenizer.supports_incremental_decode());

    const vector<Index> ids = {1, 2, 3, 4, 5, 6, 9, 0, 1};
    string concatenated;
    for (const Index id : ids)
        concatenated += tokenizer.decode_token(id);

    EXPECT_EQ(tokenizer.decode(ids), concatenated);
}

TEST(GenerationParserIncremental, IncrementalPathMatchesFullDecode)
{
    const BytePairTokenizer tokenizer = make_tokenizer();

    GenerationParserSpec spec;
    spec.initial_channel = GenerationChannel::Content;
    spec.reasoning_start = {7};
    spec.reasoning_end = {8};
    spec.stop_sequences = {{9, 10}};

    string content_deltas;
    string reasoning_deltas;
    const ChatCallback callback = [&](const ChatDelta& delta)
    {
        (delta.channel == GenerationChannel::Content
             ? content_deltas : reasoning_deltas) += delta.text;
    };

    GenerationParser parser(tokenizer, spec);

    const vector<Index> generated = {1, 2, 7, 3, 4, 8, 3, 4, 9, 10};
    for (const Index token : generated)
        if (parser.push(token, callback)) break;
    parser.finish(callback);

    EXPECT_EQ(parser.get_content(), tokenizer.decode({1, 2, 3, 4}));
    EXPECT_EQ(parser.get_reasoning(), tokenizer.decode({3, 4}));
    EXPECT_EQ(content_deltas, parser.get_content());
    EXPECT_EQ(reasoning_deltas, parser.get_reasoning());
}

TEST(GenerationParserIncremental, WithholdsIncompleteUtf8AcrossTokens)
{
    const BytePairTokenizer tokenizer = make_tokenizer();

    GenerationParserSpec spec;
    spec.initial_channel = GenerationChannel::Content;
    spec.stop_sequences = {{9, 10}};

    vector<string> deltas;
    const ChatCallback callback =
        [&](const ChatDelta& delta) { deltas.push_back(delta.text); };

    GenerationParser parser(tokenizer, spec);

    parser.push(1, callback);
    const size_t before_partial = deltas.size();
    parser.push(5, callback);                  // lone lead byte 0xC3: withheld
    EXPECT_EQ(deltas.size(), before_partial);
    parser.push(6, callback);                  // 0xA9 completes U+00E9
    parser.finish(callback);

    EXPECT_EQ(parser.get_content(), tokenizer.decode({1, 5, 6}));
    EXPECT_EQ(parser.get_content(), string("he\xC3\xA9"));

    string concatenated;
    for (const string& delta : deltas) concatenated += delta;
    EXPECT_EQ(concatenated, parser.get_content());
}
