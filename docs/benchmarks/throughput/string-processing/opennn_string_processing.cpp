#include "opennn/core/json.h"
#include "opennn/core/string_utilities.h"
#include "opennn/neural_network/operators/tokenizer_operator.h"

#include <chrono>

using namespace opennn;

int main()
{
    string corpus;
    constexpr string_view sentence =
        "OpenNN processes text, JSON, tokenizers, and datasets efficiently. ";
    corpus.reserve(4 * 1024 * 1024);
    while (corpus.size() < 4 * 1024 * 1024) corpus.append(sentence);

    auto measure = [](string_view name, auto&& operation)
    {
        const auto start = chrono::steady_clock::now();
        const size_t work = operation();
        const double seconds = chrono::duration<double>(
            chrono::steady_clock::now() - start).count();
        cout << format("{:<28} {:>12.1f} Mitems/s\n",
                       name, double(work) / seconds / 1.0e6);
    };

    measure("tokenize_views", [&]
    {
        size_t tokens = 0;
        for (int iteration = 0; iteration < 20; ++iteration)
            tokens += tokenize_views(corpus).size();
        return tokens;
    });

    WordLevelTokenizer word_level;
    word_level.set_vocabulary({
        "[PAD]", "[UNK]", "[START]", "[END]", "opennn", "processes",
        "text", "json", "tokenizers", "and", "datasets", "efficiently", ",", "."
    });

    measure("word-level encode", [&]
    {
        size_t tokens = 0;
        for (int iteration = 0; iteration < 20; ++iteration)
            tokens += word_level.encode(corpus).size();
        return tokens;
    });

    WordPieceTokenizer word_piece({
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "opennn", "processes", "text",
        "json", "token", "##izers", "and", "datasets", "efficiently", ",", "."
    });

    measure("wordpiece encode", [&]
    {
        size_t tokens = 0;
        for (int iteration = 0; iteration < 10; ++iteration)
            tokens += word_piece.encode(corpus).size();
        return tokens;
    });

    vector<string> vocabulary(20000);
    for (size_t i = 0; i < vocabulary.size(); ++i)
        vocabulary[i] = format("token_{}", i);

    const string json_text = json_array(vocabulary).dump(0);
    measure("JSON typed-array parse", [&]
    {
        size_t values = 0;
        for (int iteration = 0; iteration < 100; ++iteration)
            values += Json::parse(json_text).array_value.size();
        return values;
    });

    return 0;
}
