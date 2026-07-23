//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T O K E N I Z E R   O P E R A T O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tokenizer_operator.h"
#include "string_utilities.h"
#include "json.h"

namespace opennn
{

void TokenizerOperator::rebuild_map()
{
    vocabulary_map.clear();
    vocabulary_map.reserve(vocabulary.size());

    for (Index i = 0; i < ssize(vocabulary); ++i)
        vocabulary_map[vocabulary[size_t(i)]] = i;
}

void TokenizerOperator::set_vocabulary(const vector<string>& new_vocabulary)
{
    vocabulary = new_vocabulary;
    rebuild_map();
}

void TokenizerOperator::build_vocabulary(const vector<vector<string>>& documents,
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

Index TokenizerOperator::token_to_id(string_view token) const
{
    const auto it = vocabulary_map.find(string(token));
    return it != vocabulary_map.end() ? it->second : unk_id;
}

const string& TokenizerOperator::id_to_token(Index id) const
{
    static const string empty_token;

    if (id < 0 || id >= ssize(vocabulary))
        return empty_token;

    return vocabulary[size_t(id)];
}

vector<Index> TokenizerOperator::encode(string_view text) const
{
    const vector<string> tokens = tokenize(text);

    vector<Index> ids;
    ids.reserve(tokens.size());

    for (const string& token : tokens)
        ids.push_back(token_to_id(token));

    return ids;
}

vector<Index> TokenizerOperator::encode_sequence(const vector<string>& tokens, Index sequence_length) const
{
    vector<Index> ids;
    ids.reserve(min(size_t(sequence_length), tokens.size() + 2));
    ids.push_back(START_INDEX);

    for (const string& token : tokens)
    {
        if (ssize(ids) >= sequence_length) break;
        const auto it = vocabulary_map.find(token);
        ids.push_back(it != vocabulary_map.end() ? it->second : unk_id);
    }

    if (ssize(ids) < sequence_length)
        ids.push_back(END_INDEX);

    return ids;
}

vector<Index> TokenizerOperator::encode_sequence(string_view text, Index sequence_length) const
{
    return encode_sequence(tokenize(text), sequence_length);
}

string TokenizerOperator::decode(const vector<Index>& ids) const
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

void TokenizerOperator::to_JSON(JsonWriter& printer) const
{
    if (vocabulary.empty()) return;

    write_json(printer, {{"Vocabulary", vector_to_string(vocabulary, "\n")}});
}

void TokenizerOperator::from_JSON(const Json* element)
{
    if (element->has("Vocabulary"))
        set_vocabulary(get_tokens(read_json_string(element, "Vocabulary"), "\n"));
}

unique_ptr<TokenizerOperator> make_tokenizer_operator(const string& kind)
{
    if (kind == "WordLevel") return make_unique<WordLevelTokenizer>();
    if (kind == "WordPiece") return make_unique<WordPieceTokenizer>();
    if (kind == "BytePair")  return make_unique<BytePairTokenizer>();

    throw runtime_error("make_tokenizer_operator: unknown tokenizer kind: " + kind);
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
    return (cp >= 0x0300 && cp <= 0x036F)
        || (cp >= 0x1DC0 && cp <= 0x1DFF)
        || (cp >= 0xFE20 && cp <= 0xFE2F);
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

constexpr array<pair<uint32_t, uint32_t>, 173> case_fold_table = {{
    {0x00C0,0x0061},{0x00C1,0x0061},{0x00C2,0x0061},{0x00C3,0x0061},{0x00C4,0x0061},
    {0x00C5,0x0061},{0x00C6,0x00E6},{0x00C7,0x0063},{0x00C8,0x0065},{0x00C9,0x0065},
    {0x00CA,0x0065},{0x00CB,0x0065},{0x00CC,0x0069},{0x00CD,0x0069},{0x00CE,0x0069},
    {0x00CF,0x0069},{0x00D0,0x00F0},{0x00D1,0x006E},{0x00D2,0x006F},{0x00D3,0x006F},
    {0x00D4,0x006F},{0x00D5,0x006F},{0x00D6,0x006F},{0x00D8,0x00F8},{0x00D9,0x0075},
    {0x00DA,0x0075},{0x00DB,0x0075},{0x00DC,0x0075},{0x00DD,0x0079},{0x00DE,0x00FE},
    {0x00E0,0x0061},{0x00E1,0x0061},{0x00E2,0x0061},{0x00E3,0x0061},{0x00E4,0x0061},
    {0x00E5,0x0061},{0x00E7,0x0063},{0x00E8,0x0065},{0x00E9,0x0065},{0x00EA,0x0065},
    {0x00EB,0x0065},{0x00EC,0x0069},{0x00ED,0x0069},{0x00EE,0x0069},{0x00EF,0x0069},
    {0x00F1,0x006E},{0x00F2,0x006F},{0x00F3,0x006F},{0x00F4,0x006F},{0x00F5,0x006F},
    {0x00F6,0x006F},{0x00F9,0x0075},{0x00FA,0x0075},{0x00FB,0x0075},{0x00FC,0x0075},
    {0x00FD,0x0079},{0x00FF,0x0079},{0x0100,0x0061},{0x0101,0x0061},{0x0102,0x0061},
    {0x0103,0x0061},{0x0104,0x0061},{0x0105,0x0061},{0x0106,0x0063},{0x0107,0x0063},
    {0x0108,0x0063},{0x0109,0x0063},{0x010A,0x0063},{0x010B,0x0063},{0x010C,0x0063},
    {0x010D,0x0063},{0x010E,0x0064},{0x010F,0x0064},{0x0110,0x0111},{0x0112,0x0065},
    {0x0113,0x0065},{0x0114,0x0065},{0x0115,0x0065},{0x0116,0x0065},{0x0117,0x0065},
    {0x0118,0x0065},{0x0119,0x0065},{0x011A,0x0065},{0x011B,0x0065},{0x011C,0x0067},
    {0x011D,0x0067},{0x011E,0x0067},{0x011F,0x0067},{0x0120,0x0067},{0x0121,0x0067},
    {0x0122,0x0067},{0x0123,0x0067},{0x0124,0x0068},{0x0125,0x0068},{0x0126,0x0127},
    {0x0128,0x0069},{0x0129,0x0069},{0x012A,0x0069},{0x012B,0x0069},{0x012C,0x0069},
    {0x012D,0x0069},{0x012E,0x0069},{0x012F,0x0069},{0x0130,0x0069},{0x0132,0x0133},
    {0x0134,0x006A},{0x0135,0x006A},{0x0136,0x006B},{0x0137,0x006B},{0x0139,0x006C},
    {0x013A,0x006C},{0x013B,0x006C},{0x013C,0x006C},{0x013D,0x006C},{0x013E,0x006C},
    {0x013F,0x0140},{0x0141,0x0142},{0x0143,0x006E},{0x0144,0x006E},{0x0145,0x006E},
    {0x0146,0x006E},{0x0147,0x006E},{0x0148,0x006E},{0x014A,0x014B},{0x014C,0x006F},
    {0x014D,0x006F},{0x014E,0x006F},{0x014F,0x006F},{0x0150,0x006F},{0x0151,0x006F},
    {0x0152,0x0153},{0x0154,0x0072},{0x0155,0x0072},{0x0156,0x0072},{0x0157,0x0072},
    {0x0158,0x0072},{0x0159,0x0072},{0x015A,0x0073},{0x015B,0x0073},{0x015C,0x0073},
    {0x015D,0x0073},{0x015E,0x0073},{0x015F,0x0073},{0x0160,0x0073},{0x0161,0x0073},
    {0x0162,0x0074},{0x0163,0x0074},{0x0164,0x0074},{0x0165,0x0074},{0x0166,0x0167},
    {0x0168,0x0075},{0x0169,0x0075},{0x016A,0x0075},{0x016B,0x0075},{0x016C,0x0075},
    {0x016D,0x0075},{0x016E,0x0075},{0x016F,0x0075},{0x0170,0x0075},{0x0171,0x0075},
    {0x0172,0x0075},{0x0173,0x0075},{0x0174,0x0077},{0x0175,0x0077},{0x0176,0x0079},
    {0x0177,0x0079},{0x0178,0x0079},{0x0179,0x007A},{0x017A,0x007A},{0x017B,0x007A},
    {0x017C,0x007A},{0x017D,0x007A},{0x017E,0x007A},
}};

uint32_t fold_uncased(uint32_t cp)
{
    if (cp >= 'A' && cp <= 'Z') return cp + 32;
    if (cp < 0x00C0 || cp > 0x017E) return cp;

    const auto it = lower_bound(case_fold_table.begin(), case_fold_table.end(), cp,
                                [](const pair<uint32_t, uint32_t>& e, uint32_t value) { return e.first < value; });
    return (it != case_fold_table.end() && it->first == cp) ? it->second : cp;
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
    if (cp == 0x00D7 || cp == 0x00F7) return false;
    if (cp >= 0x00C0 && cp <= 0x024F) return true;
    if (cp >= 0x0370 && cp <= 0x03FF) return true;
    if (cp >= 0x0400 && cp <= 0x04FF) return true;
    return !is_whitespace(cp) && !is_punctuation(cp);
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
    TokenizerOperator::set_vocabulary(new_vocabulary);

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

        if (is_whitespace(codepoint)) { flush(); continue; }

        if (do_lower_case)
        {
            if (is_combining_mark(codepoint)) continue;
            codepoint = fold_uncased(codepoint);
        }

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

void WordPieceTokenizer::to_JSON(JsonWriter& printer) const
{
    TokenizerOperator::to_JSON(printer);

    write_json(printer, {{"LowerCase", to_string(do_lower_case)}});
}

void WordPieceTokenizer::from_JSON(const Json* element)
{
    if (element->has("LowerCase"))
        do_lower_case = read_json_bool(element, "LowerCase");

    TokenizerOperator::from_JSON(element);
}

BytePairTokenizer::BytePairTokenizer()
{
    reserved_tokens = {string(PAD_TOKEN)};
    unk_id = 0;

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

    vector<string> loaded_vocabulary(size_t(maximum_id + 2));
    loaded_vocabulary[0] = string(PAD_TOKEN);
    for (const auto& [token, id_value] : parsed.object_value)
        loaded_vocabulary[size_t(id_value.as_long()) + 1] = token;

    set_vocabulary(loaded_vocabulary);

    ifstream merges_file(merges_txt, ios::binary);
    throw_if(!merges_file.is_open(),
             "Cannot open merges.txt: " + merges_txt.string());

    vector<string> merge_lines;
    string line;
    while (getline(merges_file, line))
    {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        merge_lines.push_back(move(line));
    }

    set_merges(merge_lines);
}

vector<string> BytePairTokenizer::get_merges() const
{
    vector<pair<int, string>> ranked;
    ranked.reserve(merge_ranks.size());

    for (const auto& [line, rank] : merge_ranks)
        ranked.emplace_back(rank, line);

    ranges::sort(ranked);

    vector<string> merges;
    merges.reserve(ranked.size());

    for (auto& [rank, merge_line] : ranked)
        merges.push_back(move(merge_line));

    return merges;
}

void BytePairTokenizer::set_merges(const vector<string>& merges)
{
    merge_ranks.clear();
    bpe_cache.clear();

    int rank = 0;
    for (const string& merge_line : merges)
    {
        if (merge_line.empty() || merge_line[0] == '#') continue;

        if (merge_line.find(' ') == string::npos) continue;

        merge_ranks.emplace(merge_line, rank++);
    }
}

void BytePairTokenizer::to_JSON(JsonWriter& printer) const
{
    TokenizerOperator::to_JSON(printer);

    if (merge_ranks.empty()) return;

    write_json(printer, {{"Merges", vector_to_string(get_merges(), "\n")}});
}

void BytePairTokenizer::from_JSON(const Json* element)
{
    TokenizerOperator::from_JSON(element);

    if (element->has("Merges"))
        set_merges(get_tokens(read_json_string(element, "Merges"), "\n"));
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

        if (is_whitespace(c))
        {
            size_t j = i; while (j < n && is_whitespace(cps[j])) ++j;
            const size_t end = (j < n && j - i > 1) ? j - 1 : j;
            emit(i, end); i = end; continue;
        }

        emit(i, i + 1); ++i;
    }

    return pieces;
}

vector<string> BytePairTokenizer::tokenize(string_view text) const
{
    vector<string> tokens;

    for (const string& piece : pre_tokenize(text))
    {
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
        if (id == 0) continue;
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
