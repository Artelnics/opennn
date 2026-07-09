//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T O K E N I Z E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tokenizer.h"
#include "string_utilities.h"
#include "json.h"

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

bool is_ascii_digit(uint32_t cp) { return cp >= '0' && cp <= '9'; }

bool is_letter(uint32_t cp)
{
    if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) return true;
    if (cp < 0x80) return false;
    if (cp == 0x00D7 || cp == 0x00F7) return false;             // × ÷ are symbols
    if (cp >= 0x00C0 && cp <= 0x024F) return true;              // Latin supplement/extended
    if (cp >= 0x0370 && cp <= 0x03FF) return true;              // Greek
    if (cp >= 0x0400 && cp <= 0x04FF) return true;              // Cyrillic
    return !is_whitespace(cp) && !is_punctuation(cp);           // treat other non-ASCII as letters
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
    if (it == vocabulary_map.end())
        throw runtime_error("WordPieceTokenizer: vocabulary is missing " + unk_token);
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

BytePairTokenizer::BytePairTokenizer()
{
    reserved_tokens = {string(PAD_TOKEN)};
    unk_id = 0;

    // bytes_to_unicode: printable bytes map to themselves; the rest map to
    // codepoints 256.. in order, so every byte becomes one printable codepoint.
    array<bool, 256> is_direct{};
    auto mark = [&](int lo, int hi) { for (int b = lo; b <= hi; ++b) is_direct[b] = true; };
    mark('!', '~'); mark(0xA1, 0xAC); mark(0xAE, 0xFF);

    uint32_t next = 256;
    for (int b = 0; b < 256; ++b)
    {
        const uint32_t codepoint = is_direct[b] ? uint32_t(b) : next++;
        byte_encoder[size_t(b)] = codepoint;
        byte_decoder[codepoint] = static_cast<unsigned char>(b);
    }
}

void BytePairTokenizer::load(const filesystem::path& vocabulary_json,
                             const filesystem::path& merges_txt)
{
    // vocab.json: flat object { "<byte-unicode token>": id, ... }.
    ifstream vocabulary_file(vocabulary_json, ios::binary);
    throw_if(!vocabulary_file.is_open(),
             "Cannot open vocab.json: " + vocabulary_json.string());
    const string vocabulary_text((istreambuf_iterator<char>(vocabulary_file)),
                                 istreambuf_iterator<char>());

    const Json parsed = Json::parse(vocabulary_text);
    throw_if(!parsed.is_object(), "vocab.json is not a JSON object.");

    Index maximum_id = -1;
    for (const auto& [token, id_value] : parsed.object_value)
        maximum_id = max(maximum_id, Index(id_value.as_long()));

    // Reserve id 0 for [PAD]; every loaded id shifts +1.
    vector<string> loaded_vocabulary(size_t(maximum_id + 2));
    loaded_vocabulary[0] = string(PAD_TOKEN);
    for (const auto& [token, id_value] : parsed.object_value)
        loaded_vocabulary[size_t(id_value.as_long()) + 1] = token;

    set_vocabulary(loaded_vocabulary);

    // merges.txt: one "A B" pair per line, in priority order (skips the header).
    ifstream merges_file(merges_txt, ios::binary);
    throw_if(!merges_file.is_open(),
             "Cannot open merges.txt: " + merges_txt.string());

    merge_ranks.clear();
    bpe_cache.clear();
    string line;
    int rank = 0;
    while (getline(merges_file, line))
    {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;

        const size_t space = line.find(' ');
        if (space == string::npos) continue;

        merge_ranks.emplace(line, rank++);   // key = "A B" (byte-unicode tokens never contain ' ')
    }
}

vector<string> BytePairTokenizer::bpe(const string& token) const
{
    if (const auto cached = bpe_cache.find(token); cached != bpe_cache.end())
        return cached->second;

    vector<string> symbols = split_codepoints(token);

    while (symbols.size() > 1)
    {
        int best_rank = numeric_limits<int>::max();
        size_t best_index = 0;

        for (size_t i = 0; i + 1 < symbols.size(); ++i)
        {
            const auto it = merge_ranks.find(symbols[i] + ' ' + symbols[i + 1]);
            if (it != merge_ranks.end() && it->second < best_rank)
            {
                best_rank = it->second;
                best_index = i;
            }
        }

        if (best_rank == numeric_limits<int>::max()) break;

        symbols[best_index] += symbols[best_index + 1];
        symbols.erase(symbols.begin() + Index(best_index) + 1);
    }

    bpe_cache.emplace(token, symbols);
    return symbols;
}

vector<string> BytePairTokenizer::pre_tokenize(string_view text) const
{
    // Regex 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    // implemented over codepoints; a leading space attaches to the following piece.
    const vector<uint32_t> cps = utf8_to_codepoints(text);
    const size_t n = cps.size();

    vector<string> pieces;
    size_t i = 0;

    auto emit = [&](size_t start, size_t end)
    {
        string piece;
        for (size_t k = start; k < end; ++k) piece += codepoint_to_utf8(cps[k]);
        pieces.push_back(move(piece));
    };

    while (i < n)
    {
        const uint32_t c = cps[i];

        // Contractions (ASCII apostrophe only).
        if (c == '\'' && i + 1 < n)
        {
            const uint32_t d = to_lower_ascii(cps[i + 1]);
            if (d == 's' || d == 't' || d == 'm' || d == 'd') { emit(i, i + 2); i += 2; continue; }
            if (i + 2 < n)
            {
                const uint32_t e = to_lower_ascii(cps[i + 2]);
                if ((d == 'r' && e == 'e') || (d == 'v' && e == 'e') || (d == 'l' && e == 'l'))
                { emit(i, i + 3); i += 3; continue; }
            }
        }

        // Optional single leading space, then a run of letters / digits / others.
        const size_t k = (c == ' ') ? i + 1 : i;
        if (k < n && is_letter(cps[k]))
        {
            size_t j = k; while (j < n && is_letter(cps[j])) ++j;
            emit(i, j); i = j; continue;
        }
        if (k < n && is_ascii_digit(cps[k]))
        {
            size_t j = k; while (j < n && is_ascii_digit(cps[j])) ++j;
            emit(i, j); i = j; continue;
        }
        if (k < n && !is_whitespace(cps[k]) && !is_letter(cps[k]) && !is_ascii_digit(cps[k]))
        {
            size_t j = k; while (j < n && !is_whitespace(cps[j]) && !is_letter(cps[j]) && !is_ascii_digit(cps[j])) ++j;
            emit(i, j); i = j; continue;
        }

        // Whitespace run: keep the last space for the next piece's optional lead.
        if (is_whitespace(c))
        {
            size_t j = i; while (j < n && is_whitespace(cps[j])) ++j;
            const size_t end = (j < n && j - i > 1) ? j - 1 : j;
            emit(i, end); i = end; continue;
        }

        emit(i, i + 1); ++i;   // defensive fallback
    }

    return pieces;
}

vector<string> BytePairTokenizer::tokenize(string_view text) const
{
    vector<string> tokens;

    for (const string& piece : pre_tokenize(text))
    {
        // Map the piece's raw bytes through byte_encoder, then apply BPE merges.
        string byte_unicode;
        for (const char raw : piece)
            byte_unicode += codepoint_to_utf8(byte_encoder[static_cast<unsigned char>(raw)]);

        const vector<string> subwords = bpe(byte_unicode);
        tokens.insert(tokens.end(), subwords.begin(), subwords.end());
    }

    return tokens;
}

string BytePairTokenizer::decode(const vector<Index>& ids) const
{
    string byte_unicode;
    for (const Index id : ids)
    {
        if (id == 0) continue;                 // [PAD]
        byte_unicode += id_to_token(id);
    }

    string bytes;
    for (const uint32_t codepoint : utf8_to_codepoints(byte_unicode))
    {
        const auto it = byte_decoder.find(codepoint);
        if (it != byte_decoder.end()) bytes += static_cast<char>(it->second);
    }

    return bytes;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
