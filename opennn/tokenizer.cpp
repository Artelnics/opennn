//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T O K E N I Z E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tokenizer.h"
#include "string_utilities.h"

namespace opennn
{

void Tokenizer::rebuild_map()
{
    vocabulary_map.clear();
    vocabulary_map.reserve(vocabulary.size());

    for (Index i = 0; i < ssize(vocabulary); ++i)
        vocabulary_map[vocabulary[size_t(i)]] = i;
}

void Tokenizer::set_vocabulary(const vector<string>& new_vocabulary)
{
    vocabulary = new_vocabulary;
    rebuild_map();
}

void Tokenizer::build_vocabulary(const vector<vector<string>>& documents,
                                 Index maximum_vocabulary_size,
                                 Index minimum_token_frequency)
{
    unordered_map<string_view, size_t> token_count;

    for (const vector<string>& document : documents)
        for (const string& token : document)
            ++token_count[token];

    vector<pair<string_view, size_t>> sorted_tokens(token_count.begin(), token_count.end());

    ranges::sort(sorted_tokens,
                 [](const auto& a, const auto& b) { return a.second > b.second; });

    vocabulary = reserved_tokens;

    for (const auto& [token, count] : sorted_tokens)
    {
        if (count < size_t(minimum_token_frequency))
            continue;

        if (ranges::find(reserved_tokens, token) != reserved_tokens.end())
            continue;

        if (vocabulary.size() >= size_t(maximum_vocabulary_size))
            break;

        vocabulary.emplace_back(token);
    }

    rebuild_map();
}

Index Tokenizer::token_to_id(string_view token) const
{
    const auto it = vocabulary_map.find(string(token));
    return it != vocabulary_map.end() ? it->second : unk_id;
}

const string& Tokenizer::id_to_token(Index id) const
{
    static const string empty_token;

    if (id < 0 || id >= ssize(vocabulary))
        return empty_token;

    return vocabulary[size_t(id)];
}

vector<Index> Tokenizer::encode(string_view text) const
{
    const vector<string> tokens = tokenize(text);

    vector<Index> ids;
    ids.reserve(tokens.size());

    for (const string& token : tokens)
        ids.push_back(token_to_id(token));

    return ids;
}

string Tokenizer::decode(const vector<Index>& ids) const
{
    string text;

    for (const Index id : ids)
    {
        if (id == 0) continue;

        const string& token = id_to_token(id);
        if (token.empty()) continue;

        if (!text.empty()) text += ' ';
        text += token;
    }

    return text;
}

WordLevelTokenizer::WordLevelTokenizer()
{
    reserved_tokens = {"[PAD]", "[UNK]", "[START]", "[END]"};
    unk_id = 1;
}

vector<string> WordLevelTokenizer::tokenize(string_view text) const
{
    string lowered(text);
    for (char& character : lowered)
        character = static_cast<char>(tolower(static_cast<unsigned char>(character)));

    const vector<string_view> views = tokenize_views(lowered);

    vector<string> tokens;
    tokens.reserve(views.size());

    for (const string_view view : views)
        tokens.emplace_back(view);

    return tokens;
}

namespace
{

vector<uint32_t> utf8_to_codepoints(string_view text)
{
    vector<uint32_t> codepoints;
    codepoints.reserve(text.size());

    size_t i = 0;
    while (i < text.size())
    {
        const unsigned char lead = static_cast<unsigned char>(text[i]);

        uint32_t codepoint = lead;
        size_t   length    = 1;

        if      (lead < 0x80)        { codepoint = lead;        length = 1; }
        else if ((lead >> 5) == 0x6) { codepoint = lead & 0x1F; length = 2; }
        else if ((lead >> 4) == 0xE) { codepoint = lead & 0x0F; length = 3; }
        else if ((lead >> 3) == 0x1E){ codepoint = lead & 0x07; length = 4; }

        if (length == 1 || i + length > text.size())
        {
            codepoints.push_back(lead);
            ++i;
            continue;
        }

        bool valid = true;
        for (size_t k = 1; k < length; ++k)
        {
            const unsigned char continuation = static_cast<unsigned char>(text[i + k]);
            if ((continuation >> 6) != 0x2) { valid = false; break; }
            codepoint = (codepoint << 6) | (continuation & 0x3F);
        }

        if (!valid) { codepoints.push_back(lead); ++i; continue; }

        codepoints.push_back(codepoint);
        i += length;
    }

    return codepoints;
}

string codepoint_to_utf8(uint32_t codepoint)
{
    string out;
    if (codepoint < 0x80)
        out += char(codepoint);
    else if (codepoint < 0x800)
    {
        out += char(0xC0 | (codepoint >> 6));
        out += char(0x80 | (codepoint & 0x3F));
    }
    else if (codepoint < 0x10000)
    {
        out += char(0xE0 | (codepoint >> 12));
        out += char(0x80 | ((codepoint >> 6) & 0x3F));
        out += char(0x80 | (codepoint & 0x3F));
    }
    else
    {
        out += char(0xF0 | (codepoint >> 18));
        out += char(0x80 | ((codepoint >> 12) & 0x3F));
        out += char(0x80 | ((codepoint >> 6) & 0x3F));
        out += char(0x80 | (codepoint & 0x3F));
    }
    return out;
}

bool is_whitespace(uint32_t cp)
{
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x00A0;
}

bool is_control(uint32_t cp)
{
    if (cp == '\t' || cp == '\n' || cp == '\r') return false;
    return cp < 0x20 || (cp >= 0x7F && cp <= 0x9F);
}

bool is_combining_mark(uint32_t cp)
{
    return cp >= 0x0300 && cp <= 0x036F;
}

bool is_punctuation(uint32_t cp)
{
    return (cp >= 33 && cp <= 47)  || (cp >= 58 && cp <= 64)
        || (cp >= 91 && cp <= 96)  || (cp >= 123 && cp <= 126);
}

bool is_cjk(uint32_t cp)
{
    return (cp >= 0x4E00  && cp <= 0x9FFF)
        || (cp >= 0x3400  && cp <= 0x4DBF)
        || (cp >= 0x20000 && cp <= 0x2A6DF)
        || (cp >= 0x2A700 && cp <= 0x2B73F)
        || (cp >= 0x2B740 && cp <= 0x2B81F)
        || (cp >= 0x2B820 && cp <= 0x2CEAF)
        || (cp >= 0xF900  && cp <= 0xFAFF)
        || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

uint32_t to_lower_ascii(uint32_t cp)
{
    return (cp >= 'A' && cp <= 'Z') ? cp + 32 : cp;
}

vector<string> split_codepoints(string_view text)
{
    vector<string> characters;
    for (const uint32_t codepoint : utf8_to_codepoints(text))
        characters.push_back(codepoint_to_utf8(codepoint));
    return characters;
}

}

WordPieceTokenizer::WordPieceTokenizer()
{
    reserved_tokens.clear();
    unk_id = 0;
}

WordPieceTokenizer::WordPieceTokenizer(const vector<string>& new_vocabulary)
    : WordPieceTokenizer()
{
    set_vocabulary(new_vocabulary);
}

void WordPieceTokenizer::set_vocabulary(const vector<string>& new_vocabulary)
{
    Tokenizer::set_vocabulary(new_vocabulary);

    const auto it = vocabulary_map.find(unk_token);
    if (it != vocabulary_map.end())
        unk_id = it->second;
}

void WordPieceTokenizer::load_vocabulary(const filesystem::path& vocabulary_file)
{
    ifstream file(vocabulary_file);
    if (!file.is_open())
        throw runtime_error("Cannot open vocabulary file: " + vocabulary_file.string());

    vector<string> loaded_vocabulary;
    string line;

    while (getline(file, line))
    {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        loaded_vocabulary.push_back(line);
    }

    set_vocabulary(loaded_vocabulary);
}

vector<string> WordPieceTokenizer::basic_tokenize(string_view text) const
{
    vector<string> tokens;
    string current;

    auto flush = [&]()
    {
        if (!current.empty()) { tokens.push_back(current); current.clear(); }
    };

    for (uint32_t codepoint : utf8_to_codepoints(text))
    {
        if (codepoint == 0 || codepoint == 0xFFFD || is_control(codepoint)) continue;
        if (is_combining_mark(codepoint)) continue;

        if (is_whitespace(codepoint)) { flush(); continue; }

        if (do_lower_case) codepoint = to_lower_ascii(codepoint);

        if (is_punctuation(codepoint) || is_cjk(codepoint))
        {
            flush();
            tokens.push_back(codepoint_to_utf8(codepoint));
            continue;
        }

        current += codepoint_to_utf8(codepoint);
    }

    flush();
    return tokens;
}

vector<string> WordPieceTokenizer::wordpiece(const string& word) const
{
    const vector<string> characters = split_codepoints(word);

    if (Index(characters.size()) > max_input_chars_per_word)
        return { unk_token };

    vector<string> output;
    size_t start = 0;

    while (start < characters.size())
    {
        size_t end = characters.size();
        string matched;
        bool   found = false;

        while (start < end)
        {
            string substring;
            for (size_t k = start; k < end; ++k)
                substring += characters[k];

            if (start > 0) substring = continuation_prefix + substring;

            if (vocabulary_map.find(substring) != vocabulary_map.end())
            {
                matched = substring;
                found = true;
                break;
            }

            --end;
        }

        if (!found) return { unk_token };

        output.push_back(matched);
        start = end;
    }

    return output;
}

vector<string> WordPieceTokenizer::tokenize(string_view text) const
{
    vector<string> tokens;

    for (const string& word : basic_tokenize(text))
    {
        const vector<string> subwords = wordpiece(word);
        tokens.insert(tokens.end(), subwords.begin(), subwords.end());
    }

    return tokens;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
