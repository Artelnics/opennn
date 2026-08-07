//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T O K E N I Z E R   O P E R A T O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <atomic>

#include "tokenizer_operator.h"
#include "io_utilities.h"
#include "parallel_algorithms.h"
#include "string_utilities.h"
#include "json.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace opennn
{

static void hash_bytes(uint64_t& hash, string_view value)
{
    for (const unsigned char byte : value)
    {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    hash ^= 0xff;
    hash *= 1099511628211ull;
}

static void hash_number(uint64_t& hash, uint64_t value)
{
    for (int byte = 0; byte < 8; ++byte)
    {
        hash ^= (value >> (byte * 8)) & 0xff;
        hash *= 1099511628211ull;
    }
}

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

vector<string> make_vocabulary(const unordered_map<string_view, size_t>& token_count,
                               span<const string> reserved,
                               Index maximum_size,
                               Index minimum_frequency)
{
    vector<pair<string_view, size_t>> sorted_tokens(token_count.begin(), token_count.end());

    sort_parallel_if_large(
        sorted_tokens.begin(), sorted_tokens.end(),
        [](const auto& first, const auto& second)
        {
            return first.second != second.second
                ? first.second > second.second
                : first.first < second.first;
        });

    vector<string> result(reserved.begin(), reserved.end());

    for (const auto& [token, count] : sorted_tokens)
    {
        if (result.size() >= size_t(maximum_size)) break;

        if (count < size_t(minimum_frequency)
            || ranges::find(reserved, token) != reserved.end())
            continue;

        result.emplace_back(token);
    }

    return result;
}

void TokenizerOperator::build_vocabulary(const vector<vector<string>>& documents,
                                         Index maximum_vocabulary_size,
                                         Index minimum_token_frequency)
{
    unordered_map<string_view, size_t> token_count;
    const size_t tokens_number = transform_reduce(
        documents.begin(), documents.end(), size_t(0), plus<>{},
        [](const auto& document) { return document.size(); });

#ifdef _OPENMP
    if (tokens_number >= 10000)
    {
        vector<unordered_map<string_view, size_t>> local_counts(
            static_cast<size_t>(omp_get_max_threads()));

        #pragma omp parallel
        {
            auto& local = local_counts[size_t(omp_get_thread_num())];

            #pragma omp for schedule(static)
            for (Index i = 0; i < ssize(documents); ++i)
                for (const string& token : documents[size_t(i)])
                    ++local[token];
        }

        for (const auto& local : local_counts)
            for (const auto& [token, count] : local)
                token_count[token] += count;
    }
    else
#endif
    {
        for (const vector<string>& document : documents)
            for (const string& token : document)
                ++token_count[token];
    }

    vocabulary = make_vocabulary(token_count, reserved_tokens,
                                 maximum_vocabulary_size, minimum_token_frequency);
    rebuild_map();
}

uint64_t TokenizerOperator::fingerprint() const
{
    uint64_t hash = 1469598103934665603ull;
    hash_bytes(hash, get_kind());
    for (const string& token : vocabulary) hash_bytes(hash, token);
    return hash;
}

Index TokenizerOperator::token_to_id(string_view token) const
{
    if (token.data() == nullptr) return unk_id;
    const auto it = vocabulary_map.find(token);
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
    if (sequence_length <= 0) return {};

    vector<Index> ids;
    const size_t framing_tokens = size_t(start_id >= 0) + size_t(end_id >= 0);
    ids.reserve(min(size_t(sequence_length), tokens.size() + framing_tokens));

    if (start_id >= 0) ids.push_back(start_id);

    for (const string& token : tokens)
    {
        if (ssize(ids) >= sequence_length) break;
        ids.push_back(token_to_id(token));
    }

    if (end_id >= 0 && ssize(ids) < sequence_length)
        ids.push_back(end_id);

    return ids;
}

vector<Index> TokenizerOperator::encode_sequence(string_view text, Index sequence_length) const
{
    if (sequence_length <= 0) return {};

    const vector<Index> encoded = encode(text);
    vector<Index> ids;
    const size_t framing_tokens = size_t(start_id >= 0) + size_t(end_id >= 0);
    ids.reserve(min(size_t(sequence_length), encoded.size() + framing_tokens));

    if (start_id >= 0) ids.push_back(start_id);

    for (const Index id : encoded)
    {
        if (ssize(ids) >= sequence_length) break;
        ids.push_back(id);
    }

    if (end_id >= 0 && ssize(ids) < sequence_length)
        ids.push_back(end_id);

    return ids;
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

string TokenizerOperator::decode_token(Index id) const
{
    return id == 0 ? string{} : id_to_token(id);
}

void TokenizerOperator::to_JSON(JsonWriter& printer) const
{
    if (vocabulary.empty()) return;

    write_json(printer, {{"Vocabulary", json_array(vocabulary)}});
}

void TokenizerOperator::from_JSON(const Json* element)
{
    if (element->has("Vocabulary"))
        set_vocabulary(read_json_strings(element, "Vocabulary"));
}

unique_ptr<TokenizerOperator> make_tokenizer_operator(string_view kind)
{
    if (kind == "WordLevel") return make_unique<WordLevelTokenizer>();
    if (kind == "WordPiece") return make_unique<WordPieceTokenizer>();
    if (kind == "BytePair")  return make_unique<BytePairTokenizer>();
    if (kind == "Qwen3")     return make_unique<Qwen3Tokenizer>();

    throw runtime_error(format("make_tokenizer_operator: unknown tokenizer kind: {}", kind));
}

WordLevelTokenizer::WordLevelTokenizer()
    : WordLevelTokenizer({"[PAD]", "[UNK]", "[START]", "[END]"})
{
}

WordLevelTokenizer::WordLevelTokenizer(vector<string> new_reserved_tokens)
{
    reserved_tokens = move(new_reserved_tokens);

    auto resolve_id = [this](string_view token)
    {
        const auto iterator = ranges::find(reserved_tokens, token);
        return iterator == reserved_tokens.end()
            ? Index(-1)
            : Index(distance(reserved_tokens.begin(), iterator));
    };

    const Index resolved_unk_id = resolve_id("[UNK]");
    unk_id = resolved_unk_id >= 0 ? resolved_unk_id : 0;
    start_id = resolve_id("[START]");
    end_id = resolve_id("[END]");
}

vector<string> WordLevelTokenizer::tokenize(string_view text) const
{
    const string lowered = ascii_lowercase(text);
    const vector<string_view> views = tokenize_views(lowered);
    return vector<string>(views.begin(), views.end());
}

vector<Index> WordLevelTokenizer::encode(string_view text) const
{
    const string lowered = ascii_lowercase(text);

    const vector<string_view> views = tokenize_views(lowered);
    vector<Index> ids;
    ids.reserve(views.size());

    for (const string_view view : views)
        ids.push_back(token_to_id(view));

    return ids;
}

namespace
{

optional<uint32_t> next_utf8_codepoint(string_view text, size_t& position)
{
    if (position >= text.size()) return nullopt;

    const size_t start = position;
    const unsigned char lead = static_cast<unsigned char>(text[position]);
    const size_t length = utf8_sequence_length(lead);

    uint32_t codepoint = length == 1 ? lead : (lead & (0xFFu >> (length + 1)));

    if (length == 1 || start + length > text.size())
    {
        ++position;
        return codepoint;
    }

    for (size_t k = 1; k < length; ++k)
    {
        const unsigned char continuation = static_cast<unsigned char>(text[start + k]);
        if (!is_utf8_continuation(continuation))
        {
            ++position;
            return lead;
        }
        codepoint = (codepoint << 6) | (continuation & 0x3F);
    }

    position += length;
    return codepoint;
}

vector<uint32_t> utf8_to_codepoints(string_view text)
{
    vector<uint32_t> codepoints;
    codepoints.reserve(text.size());

    size_t position = 0;
    while (const optional<uint32_t> codepoint = next_utf8_codepoint(text, position))
        codepoints.push_back(*codepoint);

    return codepoints;
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
    size_t position = 0;
    while (const optional<uint32_t> codepoint = next_utf8_codepoint(text, position))
    {
        characters.emplace_back();
        append_utf8(characters.back(), *codepoint);
    }
    return characters;
}

bool is_ascii_digit(uint32_t cp) { return cp >= '0' && cp <= '9'; }

bool is_letter(uint32_t cp)
{
    if (cp < 0x80)
        return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z');

    return cp != 0x00D7 && cp != 0x00F7 && !is_whitespace(cp);
}

}

WordPieceTokenizer::WordPieceTokenizer(const vector<string>& new_vocabulary)
{
    set_vocabulary(new_vocabulary);
}

void WordPieceTokenizer::set_vocabulary(const vector<string>& new_vocabulary)
{
    TokenizerOperator::set_vocabulary(new_vocabulary);

    const auto it = vocabulary_map.find(unk_token);
    throw_if(it == vocabulary_map.end(),
             "WordPieceTokenizer: vocabulary is missing " + unk_token);
    unk_id = it->second;

    const auto cls = vocabulary_map.find("[CLS]");
    const auto sep = vocabulary_map.find("[SEP]");
    start_id = cls == vocabulary_map.end() ? -1 : cls->second;
    end_id = sep == vocabulary_map.end() ? -1 : sep->second;
}

void WordPieceTokenizer::load_vocabulary(const filesystem::path& vocabulary_file)
{
    ifstream file(vocabulary_file);
    throw_if(!file.is_open(),
             "Cannot open vocabulary file: " + vocabulary_file.string());

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

    size_t position = 0;
    while (const optional<uint32_t> parsed_codepoint = next_utf8_codepoint(text, position))
    {
        uint32_t codepoint = *parsed_codepoint;
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
            tokens.emplace_back();
            append_utf8(tokens.back(), codepoint);
            continue;
        }

        append_utf8(current, codepoint);
    }

    flush();
    return tokens;
}

void WordPieceTokenizer::wordpiece(const string& word,
                                   vector<string>* tokens,
                                   vector<Index>* ids) const
{
    vector<size_t> offsets{0};
    offsets.reserve(word.size() + 1);

    size_t position = 0;
    while (next_utf8_codepoint(word, position))
        offsets.push_back(position);

    const size_t characters = offsets.size() - 1;
    const auto append_unknown = [&]
    {
        if (tokens) tokens->push_back(unk_token);
        if (ids) ids->push_back(unk_id);
    };

    if (Index(characters) > max_input_chars_per_word)
    {
        append_unknown();
        return;
    }

    const size_t token_start = tokens ? tokens->size() : 0;
    const size_t id_start = ids ? ids->size() : 0;
    size_t start = 0;
    string candidate;
    candidate.reserve(word.size() + continuation_prefix.size());

    while (start < characters)
    {
        size_t end = characters;

        while (start < end)
        {
            candidate.clear();
            if (start > 0) candidate.append(continuation_prefix);
            candidate.append(word, offsets[start], offsets[end] - offsets[start]);

            const auto match = vocabulary_map.find(candidate);
            if (match != vocabulary_map.end())
            {
                if (tokens) tokens->push_back(candidate);
                if (ids) ids->push_back(match->second);
                break;
            }

            --end;
        }

        if (start == end)
        {
            if (tokens) tokens->resize(token_start);
            if (ids) ids->resize(id_start);
            append_unknown();
            return;
        }
        start = end;
    }
}

vector<string> WordPieceTokenizer::tokenize(string_view text) const
{
    vector<string> tokens;

    for (const string& word : basic_tokenize(text))
        wordpiece(word, &tokens, nullptr);

    return tokens;
}

vector<Index> WordPieceTokenizer::encode(string_view text) const
{
    vector<Index> ids;

    for (const string& word : basic_tokenize(text))
        wordpiece(word, nullptr, &ids);

    return ids;
}

void WordPieceTokenizer::to_JSON(JsonWriter& printer) const
{
    TokenizerOperator::to_JSON(printer);

    write_json(printer, {{"LowerCase", do_lower_case}});
}

void WordPieceTokenizer::from_JSON(const Json* element)
{
    if (element->has("LowerCase"))
        do_lower_case = read_json_bool(element, "LowerCase");

    TokenizerOperator::from_JSON(element);
}

uint64_t WordPieceTokenizer::fingerprint() const
{
    uint64_t hash = TokenizerOperator::fingerprint();
    hash_number(hash, do_lower_case);
    hash_bytes(hash, continuation_prefix);
    hash_number(hash, uint64_t(max_input_chars_per_word));
    return hash;
}

BytePairTokenizer::BytePairTokenizer()
{
    reserved_tokens = {string(PAD_TOKEN)};

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

BytePairTokenizer::BytePairTokenizer(const filesystem::path& vocabulary_json,
                                     const filesystem::path& merges_txt)
    : BytePairTokenizer()
{
    load(vocabulary_json, merges_txt);
}

void BytePairTokenizer::set_vocabulary(const vector<string>& new_vocabulary)
{
    TokenizerOperator::set_vocabulary(new_vocabulary);
    special_strings.clear();
    special_ids.clear();
}

void BytePairTokenizer::load(const filesystem::path& vocabulary_json,
                             const filesystem::path& merges_txt)
{
    const Json parsed = Json::parse(read_text_file(vocabulary_json));
    throw_if(!parsed.is_object(), "vocab.json is not a JSON object.");

    Index maximum_id = -1;
    for (const auto& [token, id_value] : parsed.object_value)
    {
        const Index id = Index(id_value.as_long());
        throw_if(id < 0, "vocab.json contains a negative token id.");
        maximum_id = max(maximum_id, id);
    }

    vector<string> loaded_vocabulary(size_t(maximum_id + 2));
    loaded_vocabulary[0] = string(PAD_TOKEN);
    for (const auto& [token, id_value] : parsed.object_value)
    {
        string& destination = loaded_vocabulary[size_t(id_value.as_long()) + 1];
        throw_if(!destination.empty(), "vocab.json contains duplicate token ids.");
        destination = token;
    }

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

    for (auto& ranked_merge : ranked)
        merges.push_back(move(ranked_merge.second));

    return merges;
}

void BytePairTokenizer::set_merges(const vector<string>& merges)
{
    merge_ranks.clear();

    int rank = 0;
    for (const string& merge_line : merges)
    {
        if (merge_line.empty() || merge_line[0] == '#'
            || merge_line.find(' ') == string::npos)
            continue;

        merge_ranks.emplace(merge_line, rank++);
    }

    static atomic<uint64_t> revision_counter{0};
    merges_revision = ++revision_counter;
}

void BytePairTokenizer::set_special_tokens(const vector<string>& new_special_tokens)
{
    special_strings.clear();
    special_ids.clear();

    for (const string& token : new_special_tokens)
    {
        const auto iterator = vocabulary_map.find(token);
        throw_if(iterator == vocabulary_map.end(),
                 "BytePairTokenizer: special token is missing from the vocabulary: " + token);

        if (special_ids.insert(iterator->second).second)
            special_strings.push_back(token);
    }

    ranges::sort(special_strings,
                 [](const string& first, const string& second)
                 {
                     return first.size() > second.size();
                 });
}

Index BytePairTokenizer::get_special_token_id(string_view token) const
{
    const auto iterator = vocabulary_map.find(token);

    return iterator != vocabulary_map.end() && special_ids.contains(iterator->second)
        ? iterator->second
        : -1;
}

void BytePairTokenizer::to_JSON(JsonWriter& printer) const
{
    TokenizerOperator::to_JSON(printer);

    if (!merge_ranks.empty())
        write_json(printer, {{"Merges", json_array(get_merges())}});
    if (!special_strings.empty())
        write_json(printer, {{"SpecialTokens", json_array(special_strings)}});
}

void BytePairTokenizer::from_JSON(const Json* element)
{
    TokenizerOperator::from_JSON(element);

    if (element->has("Merges"))
        set_merges(read_json_strings(element, "Merges"));
    set_special_tokens(read_json_strings(element, "SpecialTokens"));
}

uint64_t BytePairTokenizer::fingerprint() const
{
    uint64_t hash = TokenizerOperator::fingerprint();
    uint64_t merge_hash = 0;

    for (const auto& [pair, rank] : merge_ranks)
    {
        uint64_t entry_hash = 1469598103934665603ull;
        hash_bytes(entry_hash, pair);
        hash_number(entry_hash, uint64_t(rank));
        merge_hash ^= entry_hash;
    }

    hash_number(hash, merge_hash);
    for (const string& special : special_strings) hash_bytes(hash, special);
    return hash;
}

vector<string> BytePairTokenizer::bpe(const string& token) const
{
    vector<string> symbols = split_codepoints(token);
    string pair_key;
    pair_key.reserve(token.size() + 1);

    while (symbols.size() > 1)
    {
        int best_rank = numeric_limits<int>::max();
        size_t best_index = 0;

        for (size_t i = 0; i + 1 < symbols.size(); ++i)
        {
            pair_key.clear();
            pair_key.append(symbols[i]);
            pair_key.push_back(' ');
            pair_key.append(symbols[i + 1]);

            const auto it = merge_ranks.find(pair_key);
            if (it == merge_ranks.end() || it->second >= best_rank) continue;

            best_rank = it->second;
            best_index = i;
        }

        if (best_rank == numeric_limits<int>::max()) break;

        symbols[best_index] += symbols[best_index + 1];
        symbols.erase(symbols.begin() + Index(best_index) + 1);
    }

    return symbols;
}

namespace
{

struct PreTokenizeRun
{
    explicit PreTokenizeRun(string_view text)
        : cps(utf8_to_codepoints(text))
    {
    }

    void emit(size_t start, size_t end)
    {
        string piece;
        piece.reserve((end - start) * 2);
        for (size_t k = start; k < end; ++k) append_utf8(piece, cps[k]);
        pieces.push_back(move(piece));
    }

    bool try_contraction(size_t& i)
    {
        if (cps[i] != '\'' || i + 1 >= cps.size()) return false;
        const uint32_t d = to_lower_ascii(cps[i + 1]);
        if (d == 's' || d == 't' || d == 'm' || d == 'd') { emit(i, i + 2); i += 2; return true; }
        if (i + 2 < cps.size())
        {
            const uint32_t e = to_lower_ascii(cps[i + 2]);
            if ((d == 'r' && e == 'e') || (d == 'v' && e == 'e') || (d == 'l' && e == 'l'))
            { emit(i, i + 3); i += 3; return true; }
        }
        return false;
    }

    void emit_single(size_t& i) { emit(i, i + 1); ++i; }

    vector<uint32_t> cps;
    vector<string> pieces;
};

}

vector<string> BytePairTokenizer::pre_tokenize(string_view text) const
{
    PreTokenizeRun run(text);
    const vector<uint32_t>& cps = run.cps;
    const size_t n = cps.size();
    size_t i = 0;

    while (i < n)
    {
        const uint32_t c = cps[i];

        if (run.try_contraction(i)) continue;

        const size_t k = (c == ' ') ? i + 1 : i;
        if (k < n && is_letter(cps[k]))
        {
            size_t j = k; while (j < n && is_letter(cps[j])) ++j;
            run.emit(i, j); i = j; continue;
        }
        if (k < n && is_ascii_digit(cps[k]))
        {
            size_t j = k; while (j < n && is_ascii_digit(cps[j])) ++j;
            run.emit(i, j); i = j; continue;
        }
        if (k < n && !is_whitespace(cps[k]) && !is_letter(cps[k]) && !is_ascii_digit(cps[k]))
        {
            size_t j = k; while (j < n && !is_whitespace(cps[j]) && !is_letter(cps[j]) && !is_ascii_digit(cps[j])) ++j;
            run.emit(i, j); i = j; continue;
        }

        if (is_whitespace(c))
        {
            size_t j = i; while (j < n && is_whitespace(cps[j])) ++j;
            const size_t end = (j < n && j - i > 1) ? j - 1 : j;
            run.emit(i, end); i = end; continue;
        }

        run.emit_single(i);
    }

    return move(run.pieces);
}

void BytePairTokenizer::tokenize_into(string_view text,
                                      vector<string>* tokens,
                                      vector<Index>* ids) const
{
    const auto append = [&](const string& token)
    {
        if (tokens) tokens->push_back(token);
        if (ids) ids->push_back(token_to_id(token));
    };

    constexpr size_t maximum_cache_entries = 4096;
    static thread_local unordered_map<uint64_t, StringMap<vector<string>>> caches;
    if (caches.size() > 8 && !caches.contains(merges_revision))
        caches.clear();
    StringMap<vector<string>>& cache = caches[merges_revision];

    auto append_segment = [&](string_view segment)
    {
        for (const string& piece : pre_tokenize(segment))
        {
            string byte_unicode;
            byte_unicode.reserve(piece.size() * 2);
            for (const char raw : piece)
                append_utf8(byte_unicode, byte_encoder[static_cast<unsigned char>(raw)]);

            const vector<string>* subwords = nullptr;
            vector<string> uncached;

            const auto cached = cache.find(byte_unicode);
            if (cached != cache.end())
            {
                subwords = &cached->second;
            }
            else if (cache.size() < maximum_cache_entries)
            {
                auto iterator = cache.try_emplace(byte_unicode).first;
                iterator->second = bpe(iterator->first);
                subwords = &iterator->second;
            }
            else
            {
                uncached = bpe(byte_unicode);
                subwords = &uncached;
            }

            for (const string& subword : *subwords)
                append(subword);
        }
    };

    size_t position = 0;
    while (position < text.size())
    {
        size_t special_position = string::npos;
        const string* matched_special = nullptr;

        for (const string& special : special_strings)
        {
            const size_t found = text.find(special, position);
            if (found >= special_position) continue;

            special_position = found;
            matched_special = &special;
        }

        const size_t segment_end = min(special_position, text.size());

        if (segment_end > position)
            append_segment(text.substr(position, segment_end - position));
        if (!matched_special)
            break;

        append(*matched_special);
        position = special_position + matched_special->size();
    }
}

vector<string> BytePairTokenizer::tokenize(string_view text) const
{
    vector<string> tokens;
    tokenize_into(text, &tokens, nullptr);
    return tokens;
}

vector<Index> BytePairTokenizer::encode(string_view text) const
{
    vector<Index> ids;
    tokenize_into(text, nullptr, &ids);
    return ids;
}

string BytePairTokenizer::decode(const vector<Index>& ids) const
{
    string bytes;
    for (const Index id : ids)
        bytes += decode_token(id);

    return bytes;
}

string BytePairTokenizer::decode_token(Index id) const
{
    if (id == 0 || special_ids.contains(id)) return {};

    string bytes;
    const string& token = id_to_token(id);
    size_t position = 0;
    while (const optional<uint32_t> codepoint = next_utf8_codepoint(token, position))
    {
        const auto it = byte_decoder.find(*codepoint);
        if (it != byte_decoder.end()) bytes.push_back(static_cast<char>(it->second));
    }

    return bytes;
}

Qwen3Tokenizer::Qwen3Tokenizer(const filesystem::path& directory)
{
    load(directory / "vocab.json",
         directory / "merges.txt",
         directory / "qwen3_special.tsv");
}

void Qwen3Tokenizer::load(const filesystem::path& vocabulary_json,
                          const filesystem::path& merges_txt,
                          const filesystem::path& special_tokens_tsv)
{
    BytePairTokenizer::load(vocabulary_json, merges_txt);

    ifstream special_file(special_tokens_tsv, ios::binary);
    throw_if(!special_file.is_open(), "Cannot open special tokens: " + special_tokens_tsv.string());

    vector<string> loaded_specials;
    string line;

    while (getline(special_file, line))
    {
        if (!line.empty() && line.back() == '\r') line.pop_back();

        const size_t tab = line.find('\t');
        if (tab == string::npos) continue;

        const string_view line_view = line;
        const Index id = parse_number<Index>(
            line_view.substr(0, tab), "Qwen3 special-token id", "integer");
        const string token(line_view.substr(tab + 1));
        if (token.empty()) continue;

        throw_if(id < 0, "Special-token file contains a negative token id.");

        const size_t shifted_id = size_t(id) + 1;
        if (shifted_id >= vocabulary.size())
            vocabulary.resize(shifted_id + 1);

        string& destination = vocabulary[shifted_id];
        throw_if(!destination.empty() && destination != token,
                 "Special-token id collides with the base vocabulary.");

        destination = token;
        loaded_specials.push_back(token);
    }

    rebuild_map();
    set_special_tokens(loaded_specials);
}

vector<string> Qwen3Tokenizer::pre_tokenize(string_view text) const
{
    PreTokenizeRun run(text);
    const vector<uint32_t>& cps = run.cps;
    const size_t n = cps.size();
    size_t i = 0;

    auto is_crlf  = [](uint32_t c) { return c == '\r' || c == '\n'; };
    auto is_other = [](uint32_t c) { return !is_whitespace(c) && !is_letter(c) && !is_ascii_digit(c); };

    while (i < n)
    {
        const uint32_t c = cps[i];

        if (run.try_contraction(i)) continue;

        size_t run_start = string::npos;
        if (is_letter(c))
            run_start = i;
        else if (!is_crlf(c) && !is_ascii_digit(c)
                 && i + 1 < n && is_letter(cps[i + 1]))
            run_start = i + 1;

        if (run_start != string::npos)
        {
            size_t j = run_start;
            while (j < n && is_letter(cps[j])) ++j;
            run.emit(i, j);
            i = j;
            continue;
        }

        if (is_ascii_digit(c))
        {
            run.emit_single(i);
            continue;
        }

        const size_t other_start = c == ' ' ? i + 1 : i;
        if (other_start < n && is_other(cps[other_start]))
        {
            size_t j = other_start;
            while (j < n && is_other(cps[j])) ++j;
            while (j < n && is_crlf(cps[j])) ++j;
            run.emit(i, j);
            i = j;
            continue;
        }

        if (is_whitespace(c))
        {
            size_t j = i; while (j < n && is_whitespace(cps[j])) ++j;
            size_t newline_end = j;
            while (newline_end > i && !is_crlf(cps[newline_end - 1]))
                --newline_end;

            if (newline_end > i)
            {
                run.emit(i, newline_end);
                i = newline_end;
                continue;
            }
            const size_t end = (j < n && j - i > 1) ? j - 1 : j;
            run.emit(i, end); i = end; continue;
        }

        run.emit_single(i);
    }

    return move(run.pieces);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
