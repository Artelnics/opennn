#include "tests/pch.h"

#include "opennn/neural_network/operators/tokenizer_operator.h"

using namespace opennn;

namespace
{
    const vector<string> wordpiece_vocabulary = {
        "[PAD]",
        "[UNK]",
        "play",
        "##ing",
        "##ed",
        "un",
        "##aff",
        "##able",
        "love",
        "i",
        ",",
        "!"
    };

    string write_temp_vocabulary(const string& name, const vector<string>& tokens)
    {
        const string file_path = (filesystem::temp_directory_path() / name).string();

        ofstream file(file_path);
        for (const string& token : tokens)
            file << token << "\n";
        file.close();

        return file_path;
    }

    void write_file(const filesystem::path& path, string_view contents)
    {
        ofstream file(path, ios::binary);
        file << contents;
    }

    filesystem::path make_qwen_tokenizer_directory()
    {
        const filesystem::path directory =
            filesystem::temp_directory_path() / "opennn_qwen3_tokenizer_test";

        filesystem::create_directories(directory);

        write_file(directory / "vocab.json",
                   R"({"h":0,"i":1,"hi":2,"1":3,"2":4,"12":5})");
        write_file(directory / "merges.txt",
                   "#version: 0.2\nh i\n1 2\n");
        write_file(directory / "qwen3_special.tsv",
                   "6\t<|im_start|>\n7\t<|im_end|>\n8\t<|endoftext|>\n");

        return directory;
    }
}

TEST(WordLevelTokenizer, TokenizeLowercasesAndSeparatesWordsAndPunctuation)
{
    WordLevelTokenizer tokenizer;

    EXPECT_EQ(tokenizer.tokenize("The quick, brown Fox!"),
              (vector<string>{"the", "quick", ",", "brown", "fox", "!"}));
}

TEST(WordLevelTokenizer, BuildVocabularyReservedFirstThenByFrequency)
{
    WordLevelTokenizer tokenizer;

    const vector<vector<string>> documents = { {"a", "b", "a"}, {"a", "c"} };
    tokenizer.build_vocabulary(documents, 20000, 1);

    const vector<string>& vocabulary = tokenizer.get_vocabulary();

    ASSERT_EQ(tokenizer.get_vocabulary_size(), 7);
    EXPECT_EQ(vocabulary[0], "[PAD]");
    EXPECT_EQ(vocabulary[1], "[UNK]");
    EXPECT_EQ(vocabulary[2], "[START]");
    EXPECT_EQ(vocabulary[3], "[END]");
    EXPECT_EQ(vocabulary[4], "a");
    EXPECT_EQ(vocabulary[5], "b");
    EXPECT_EQ(vocabulary[6], "c");
    EXPECT_EQ(tokenizer.token_to_id("a"), 4);

    EXPECT_NE(tokenizer.token_to_id("b"), tokenizer.get_unk_id());
    EXPECT_NE(tokenizer.token_to_id("c"), tokenizer.get_unk_id());
}

TEST(WordLevelTokenizer, BuildVocabularyRespectsMinimumFrequency)
{
    WordLevelTokenizer tokenizer;

    const vector<vector<string>> documents = { {"a", "b", "a"}, {"a", "c"} };
    tokenizer.build_vocabulary(documents, 20000, 2);

    EXPECT_EQ(tokenizer.get_vocabulary_size(), 5);
    EXPECT_EQ(tokenizer.token_to_id("a"), 4);
    EXPECT_EQ(tokenizer.token_to_id("b"), tokenizer.get_unk_id());
}

TEST(WordLevelTokenizer, BuildVocabularyRespectsMaximumSize)
{
    WordLevelTokenizer tokenizer;

    const vector<vector<string>> documents = { {"a", "b", "a"}, {"a", "c"} };
    tokenizer.build_vocabulary(documents, 5, 1);

    EXPECT_EQ(tokenizer.get_vocabulary_size(), 5);
    EXPECT_EQ(tokenizer.token_to_id("a"), 4);
}

TEST(WordLevelTokenizer, TokenToIdReturnsUnkForUnknownTokens)
{
    WordLevelTokenizer tokenizer;

    const vector<vector<string>> documents = { {"a", "b"} };
    tokenizer.build_vocabulary(documents, 20000, 1);

    EXPECT_EQ(tokenizer.get_unk_id(), 1);
    EXPECT_EQ(tokenizer.token_to_id("not_in_vocabulary"), 1);
}

TEST(WordLevelTokenizer, EncodeAndDecodeRoundTrip)
{
    WordLevelTokenizer tokenizer;
    tokenizer.set_vocabulary({"[PAD]", "[UNK]", "hello", "world"});

    const vector<Index> ids = tokenizer.encode("Hello World");
    EXPECT_EQ(ids, (vector<Index>{2, 3}));

    EXPECT_EQ(tokenizer.decode(ids), "hello world");
    EXPECT_EQ(tokenizer.decode({0, 2, 0, 3}), "hello world");
}

TEST(WordLevelTokenizer, EncodeSequenceAddsConfiguredFraming)
{
    WordLevelTokenizer tokenizer;
    tokenizer.set_vocabulary({"[PAD]", "[UNK]", "[START]", "[END]", "hello"});

    EXPECT_EQ(tokenizer.encode_sequence("hello", 4), (vector<Index>{2, 4, 3}));
}

TEST(WordPieceTokenizer, GreedyLongestMatchSubwords)
{
    WordPieceTokenizer tokenizer(wordpiece_vocabulary);

    EXPECT_EQ(tokenizer.tokenize("playing"),   (vector<string>{"play", "##ing"}));
    EXPECT_EQ(tokenizer.tokenize("played"),    (vector<string>{"play", "##ed"}));
    EXPECT_EQ(tokenizer.tokenize("unaffable"), (vector<string>{"un", "##aff", "##able"}));
}

TEST(WordPieceTokenizer, UnknownWordBecomesSingleUnk)
{
    WordPieceTokenizer tokenizer(wordpiece_vocabulary);

    EXPECT_EQ(tokenizer.get_unk_id(), 1);
    EXPECT_EQ(tokenizer.tokenize("xyz"), (vector<string>{"[UNK]"}));
    EXPECT_EQ(tokenizer.token_to_id("xyz"), 1);
}

TEST(WordPieceTokenizer, LowercasesAndSplitsPunctuation)
{
    WordPieceTokenizer tokenizer(wordpiece_vocabulary);

    EXPECT_EQ(tokenizer.tokenize("I love, playing!"),
              (vector<string>{"i", "love", ",", "play", "##ing", "!"}));
}

TEST(WordPieceTokenizer, StripsAccentsAndLowercasesUnicode)
{
    WordPieceTokenizer tokenizer({"[UNK]", "cafe", "nino", "hello"});

    EXPECT_EQ(tokenizer.tokenize("Café"),  (vector<string>{"cafe"}));
    EXPECT_EQ(tokenizer.tokenize("NIÑO"),  (vector<string>{"nino"}));
    EXPECT_EQ(tokenizer.tokenize("Héllo"), (vector<string>{"hello"}));
}

TEST(WordPieceTokenizer, EncodeMapsSubwordsToIds)
{
    WordPieceTokenizer tokenizer(wordpiece_vocabulary);

    EXPECT_EQ(tokenizer.encode("playing"), (vector<Index>{2, 3}));
}

TEST(WordPieceTokenizer, EncodeSequenceUsesClsAndSepWhenAvailable)
{
    WordPieceTokenizer tokenizer(
        {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "play", "##ing"});

    EXPECT_EQ(tokenizer.encode_sequence("playing", 6),
              (vector<Index>{2, 4, 5, 3}));
    EXPECT_EQ(tokenizer.encode_sequence("playing", 3),
              (vector<Index>{2, 4, 5}));
}

TEST(WordPieceTokenizer, SetVocabularyResolvesUnkId)
{
    WordPieceTokenizer tokenizer;
    tokenizer.set_vocabulary({"[PAD]", "token", "[UNK]", "x"});

    EXPECT_EQ(tokenizer.get_unk_id(), 2);
    EXPECT_EQ(tokenizer.token_to_id("missing"), 2);
}

TEST(WordPieceTokenizer, BuildVocabularyIsNoOpForFixedVocabulary)
{
    WordPieceTokenizer tokenizer(wordpiece_vocabulary);

    tokenizer.build_vocabulary({ {"foo", "bar"} }, 10, 1);

    EXPECT_EQ(tokenizer.get_vocabulary_size(), Index(wordpiece_vocabulary.size()));
    EXPECT_EQ(tokenizer.tokenize("playing"), (vector<string>{"play", "##ing"}));
}

TEST(WordPieceTokenizer, CasedModeKeepsOriginalCase)
{
    WordPieceTokenizer tokenizer({"[UNK]", "Play"});
    tokenizer.set_lower_case(false);

    EXPECT_EQ(tokenizer.tokenize("Play"), (vector<string>{"Play"}));
    EXPECT_EQ(tokenizer.tokenize("play"), (vector<string>{"[UNK]"}));
}

TEST(WordPieceTokenizer, LoadVocabularyFromFile)
{
    const string vocabulary_path = write_temp_vocabulary("opennn_wordpiece_vocab.txt", wordpiece_vocabulary);

    WordPieceTokenizer tokenizer;
    tokenizer.load_vocabulary(vocabulary_path);

    EXPECT_EQ(tokenizer.get_vocabulary_size(), Index(wordpiece_vocabulary.size()));
    EXPECT_EQ(tokenizer.get_unk_id(), 1);
    EXPECT_EQ(tokenizer.tokenize("playing"), (vector<string>{"play", "##ing"}));

    error_code error;
    filesystem::remove(vocabulary_path, error);
}

TEST(BytePairTokenizer, EncodeSequenceDoesNotAssumeWordLevelFraming)
{
    BytePairTokenizer tokenizer;
    tokenizer.set_vocabulary({"[PAD]", "a", "b"});

    EXPECT_EQ(tokenizer.encode_sequence(vector<string>{"a", "b"}, 4),
              (vector<Index>{1, 2}));
}

TEST(Qwen3Tokenizer, LoadsSpecialTokensAndPreservesThemWhenCloned)
{
    const filesystem::path directory = make_qwen_tokenizer_directory();
    Qwen3Tokenizer tokenizer(directory);

    EXPECT_EQ(tokenizer.get_im_start_id(), 7);
    EXPECT_EQ(tokenizer.get_im_end_id(), 8);
    EXPECT_EQ(tokenizer.get_endoftext_id(), 9);
    EXPECT_EQ(tokenizer.tokenize("hi<|im_end|>"),
              (vector<string>{"hi", "<|im_end|>"}));
    EXPECT_EQ(tokenizer.encode("hi<|im_end|>"), (vector<Index>{3, 8}));
    EXPECT_EQ(tokenizer.decode({3, 8}), "hi");

    const unique_ptr<TokenizerOperator> cloned = tokenizer.clone();
    const auto* qwen_clone = dynamic_cast<const Qwen3Tokenizer*>(cloned.get());

    ASSERT_NE(qwen_clone, nullptr);
    EXPECT_EQ(qwen_clone->get_im_end_id(), 8);
    EXPECT_EQ(qwen_clone->encode("hi<|im_end|>"), (vector<Index>{3, 8}));

    error_code error;
    filesystem::remove_all(directory, error);
}

TEST(Qwen3Tokenizer, KeepsDigitsAsSeparatePretokens)
{
    const filesystem::path directory = make_qwen_tokenizer_directory();

    BytePairTokenizer gpt_tokenizer(directory / "vocab.json",
                                    directory / "merges.txt");
    Qwen3Tokenizer qwen_tokenizer(directory);

    EXPECT_EQ(gpt_tokenizer.encode("12"), (vector<Index>{6}));
    EXPECT_EQ(qwen_tokenizer.encode("12"), (vector<Index>{4, 5}));

    error_code error;
    filesystem::remove_all(directory, error);
}
