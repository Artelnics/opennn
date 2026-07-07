#include "pch.h"

#include "../opennn/tokenizer.h"

using namespace opennn;

namespace
{
    const vector<string> wordpiece_vocabulary = {
        "[PAD]",   // 0
        "[UNK]",   // 1
        "play",    // 2
        "##ing",   // 3
        "##ed",    // 4
        "un",      // 5
        "##aff",   // 6
        "##able",  // 7
        "love",    // 8
        "i",       // 9
        ",",       // 10
        "!"        // 11
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
}


// ------------------------- WordLevelTokenizer -------------------------

TEST(WordLevelTokenizer, TokenizeLowercasesAndSeparatesWordsAndPunctuation)
{
    WordLevelTokenizer tokenizer;

    EXPECT_EQ(tokenizer.tokenize("The quick, brown Fox!"),
              (vector<string>{"the", "quick", ",", "brown", "fox", "!"}));
}

TEST(WordLevelTokenizer, BuildVocabularyReservedFirstThenByFrequency)
{
    WordLevelTokenizer tokenizer;

    // a:3, b:1, c:1
    const vector<vector<string>> documents = { {"a", "b", "a"}, {"a", "c"} };
    tokenizer.build_vocabulary(documents, 20000, 1);

    const vector<string>& vocabulary = tokenizer.get_vocabulary();

    ASSERT_EQ(tokenizer.get_vocabulary_size(), 7);   // 4 reserved + a + b + c
    EXPECT_EQ(vocabulary[0], "[PAD]");
    EXPECT_EQ(vocabulary[1], "[UNK]");
    EXPECT_EQ(vocabulary[2], "[START]");
    EXPECT_EQ(vocabulary[3], "[END]");
    EXPECT_EQ(vocabulary[4], "a");                   // strictly most frequent, right after reserved
    EXPECT_EQ(tokenizer.token_to_id("a"), 4);

    EXPECT_NE(tokenizer.token_to_id("b"), tokenizer.get_unk_id());
    EXPECT_NE(tokenizer.token_to_id("c"), tokenizer.get_unk_id());
}

TEST(WordLevelTokenizer, BuildVocabularyRespectsMinimumFrequency)
{
    WordLevelTokenizer tokenizer;

    const vector<vector<string>> documents = { {"a", "b", "a"}, {"a", "c"} };
    tokenizer.build_vocabulary(documents, 20000, 2);   // only "a" reaches frequency 2

    EXPECT_EQ(tokenizer.get_vocabulary_size(), 5);      // 4 reserved + a
    EXPECT_EQ(tokenizer.token_to_id("a"), 4);
    EXPECT_EQ(tokenizer.token_to_id("b"), tokenizer.get_unk_id());
}

TEST(WordLevelTokenizer, BuildVocabularyRespectsMaximumSize)
{
    WordLevelTokenizer tokenizer;

    const vector<vector<string>> documents = { {"a", "b", "a"}, {"a", "c"} };
    tokenizer.build_vocabulary(documents, 5, 1);        // max size counts reserved tokens

    EXPECT_EQ(tokenizer.get_vocabulary_size(), 5);      // 4 reserved + the single most frequent
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
    EXPECT_EQ(tokenizer.decode({0, 2, 0, 3}), "hello world");   // padding (id 0) is skipped
}


// ------------------------- WordPieceTokenizer -------------------------

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

TEST(WordPieceTokenizer, EncodeMapsSubwordsToIds)
{
    WordPieceTokenizer tokenizer(wordpiece_vocabulary);

    EXPECT_EQ(tokenizer.encode("playing"), (vector<Index>{2, 3}));
}

TEST(WordPieceTokenizer, SetVocabularyResolvesUnkId)
{
    WordPieceTokenizer tokenizer;
    tokenizer.set_vocabulary({"[PAD]", "token", "[UNK]", "x"});   // [UNK] at index 2

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
    EXPECT_EQ(tokenizer.tokenize("play"), (vector<string>{"[UNK]"}));   // lowercase not in vocab
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
