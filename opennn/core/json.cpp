// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#include "opennn/core/json.h"
#include "opennn/core/io_utilities.h"
#include "opennn/core/string_utilities.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#if defined(__APPLE__) && defined(_LIBCPP_VERSION)
#include <locale>
#include <sstream>
#endif

namespace opennn
{

namespace
{

bool parse_double_exact(std::string_view text, double& value)
{
#if defined(__APPLE__) && defined(_LIBCPP_VERSION)
    // Xcode 16's libc++ declares floating-point from_chars but does not
    // implement it. The classic locale preserves JSON's decimal grammar.
    std::istringstream stream{std::string(text)};
    stream.imbue(std::locale::classic());
    stream >> std::noskipws >> value;
    return stream.eof() && !stream.fail();
#else
    const char* const first = text.data();
    const char* const last = first + text.size();
    const auto [end, error] = std::from_chars(first, last, value);
    return error == std::errc{} && end == last;
#endif
}

}

Json Json::make_object()
{
    Json json;
    json.value.emplace<Object>();
    return json;
}

Json Json::make_array()
{
    Json json;
    json.value.emplace<Array>();
    return json;
}

Json::Array& Json::as_array()
{
    throw_if(!is_array(), "JSON: value is not an array");
    return std::get<Array>(value);
}

const Json::Array& Json::as_array() const
{
    throw_if(!is_array(), "JSON: value is not an array");
    return std::get<Array>(value);
}

Json::Object& Json::as_object()
{
    throw_if(!is_object(), "JSON: value is not an object");
    return std::get<Object>(value);
}

const Json::Object& Json::as_object() const
{
    throw_if(!is_object(), "JSON: value is not an object");
    return std::get<Object>(value);
}

const Json* Json::find(std::string_view key) const
{
    if (!is_object()) return nullptr;
    const Object& object = as_object();
    const auto it = std::ranges::find_if(object,
                                         [key](const auto& item) { return item.first == key; });
    return it != object.end() ? &it->second : nullptr;
}

const Json& Json::at(std::string_view key) const
{
    const Json* const v = find(key);
    throw_if(!v, "JSON: missing key '{}'", key);
    return *v;
}

Json& Json::operator[](std::string_view key)
{
    if (!is_object()) value.emplace<Object>();
    Object& object = as_object();
    for (auto& [k, v] : object)
        if (k == key) return v;
    object.emplace_back(std::string(key), Json{});
    return object.back().second;
}

Json& Json::set(std::string_view key, Json new_value)
{
    (*this)[key] = std::move(new_value);
    return *this;
}

void Json::push_back(Json new_value)
{
    if (!is_array()) this->value.emplace<Array>();
    as_array().push_back(std::move(new_value));
}

std::string Json::as_string() const
{
    using enum Kind;
    switch (get_kind())
    {
    case Null:   return "";
    case Bool:   return std::get<bool>(value) ? "1" : "0";
    case Number: return std::format("{:.10g}", std::get<double>(value));
    case String: return std::get<std::string>(value);
    case Array:
    case Object: return dump(0);
    }

    throw std::runtime_error("JSON: invalid value kind");
}

long long Json::as_long() const
{
    using enum Kind;
    switch (get_kind())
    {
    case Number:
    {
        const double number = std::get<double>(value);
        constexpr long long minimum = std::numeric_limits<long long>::min();
        constexpr long long maximum = std::numeric_limits<long long>::max();
        throw_if(!std::isfinite(number) || number < double(minimum) || number > double(maximum),
                 "JSON: numeric value is outside the integer range");
        // JSON numbers use double storage. The maximum integer rounds up to
        // 2^63, including existing saved optimizer 'unlimited' settings.
        // Recover that boundary without an out-of-range floating-point cast.
        if (number == double(maximum)) return maximum;
        return static_cast<long long>(number);
    }
    case Bool:   return std::get<bool>(value) ? 1 : 0;
    case String:
    {
        const std::string& string = std::get<std::string>(value);
        if (string.empty()) return 0LL;
        return parse_number<long long>(string, "JSON", "integer");
    }
    case Null:
    case Array:
    case Object: return 0;
    }

    throw std::runtime_error("JSON: invalid value kind");
}

double Json::as_double() const
{
    using enum Kind;
    switch (get_kind())
    {
    case Number: return std::get<double>(value);
    case Bool:   return std::get<bool>(value) ? 1.0 : 0.0;
    case String: {
        const std::string& string = std::get<std::string>(value);
        if (string.empty()) return 0.0;
        double number = 0.0;
        throw_if(!parse_double_exact(string, number),
                 "JSON: invalid numeric value '{}'", string);
        return number;
    }
    case Null:
    case Array:
    case Object: return 0.0;
    }

    throw std::runtime_error("JSON: invalid value kind");
}

bool Json::as_bool() const
{
    using enum Kind;
    switch (get_kind())
    {
    case Bool:   return std::get<bool>(value);
    case Number: return std::get<double>(value) != 0.0;
    case String: return contains({"1", "true"}, std::get<std::string>(value));
    case Null:
    case Array:
    case Object: return false;
    }

    throw std::runtime_error("JSON: invalid value kind");
}
static void escape_string(std::string& out, const std::string& s)
{
    out.push_back('"');
    for (const char c : s)
    {
        switch (c)
        {
        case '"':  out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n";  break;
        case '\r': out += "\\r";  break;
        case '\t': out += "\\t";  break;
        case '\b': out += "\\b";  break;
        case '\f': out += "\\f";  break;
        default:
            if (static_cast<unsigned char>(c) < 0x20)
            {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                out += buf;
            }
            else out.push_back(c);
        }
    }
    out.push_back('"');
}

static void dump_value(std::string& out, const Json& v, int indent, int depth);

static void dump_indent(std::string& out, int indent, int depth)
{
    if (indent <= 0) return;
    out.push_back('\n');
    for (int i = 0; i < indent * depth; ++i) out.push_back(' ');
}

static void dump_value(std::string& out, const Json& v, int indent, int depth)
{
    using enum Json::Kind;
    switch (v.get_kind())
    {
    case Null:   out += "null"; return;
    case Bool:   out += (v.as_bool() ? "true" : "false"); return;
    case Number: {
        const double number = v.as_double();
        char buf[32];

        if (!std::isfinite(number)) { out += "null"; return; }

        if (std::abs(number) < 1e15 && number == std::trunc(number))
            std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(number));
        else
        {
            auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf) - 1, number);
            *ptr = '\0';
        }
        out += buf;
        return;
    }
    case String: escape_string(out, v.as_string()); return;
    case Array:
    {
        const Json::Array& array = v.as_array();
        if (array.empty()) { out += "[]"; return; }
        out.push_back('[');
        for (std::size_t i = 0; i < array.size(); ++i)
        {
            dump_indent(out, indent, depth + 1);
            dump_value(out, array[i], indent, depth + 1);
            if (i + 1 < array.size()) out.push_back(',');
        }
        dump_indent(out, indent, depth);
        return out.push_back(']');
    }
    case Object:
    {
        const Json::Object& object = v.as_object();
        if (object.empty()) { out += "{}"; return; }
        out.push_back('{');
        for (std::size_t i = 0; i < object.size(); ++i)
        {
            dump_indent(out, indent, depth + 1);
            escape_string(out, object[i].first);
            out += indent > 0 ? ": " : ":";
            dump_value(out, object[i].second, indent, depth + 1);
            if (i + 1 < object.size()) out.push_back(',');
        }
        dump_indent(out, indent, depth);
        return out.push_back('}');
    }
    }
}

std::string Json::dump(int indent) const
{
    std::string out;
    dump_value(out, *this, indent, 0);
    return out;
}

namespace {

constexpr std::size_t max_json_input_bytes = std::size_t(256) * 1024 * 1024;

struct Parser
{
    static constexpr std::size_t max_nesting_depth = 256;

    std::string_view s;
    std::size_t position = 0;

    explicit Parser(std::string_view text) : s(text) {}

    void skip_ws()
    {
        while (position < s.size())
        {
            const char c = s[position];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++position;
            else break;
        }
    }

    [[noreturn]] void fail(const std::string& msg) const
    {
        throw std::runtime_error(std::format("JSON parse error at {}: {}", position, msg));
    }

    char peek()
    {
        skip_ws();
        if (position >= s.size()) fail("unexpected end of input");
        return s[position];
    }

    char consume()
    {
        skip_ws();
        if (position >= s.size()) fail("unexpected end of input");
        return s[position++];
    }

    bool match(std::string_view word)
    {
        skip_ws();
        const std::size_t n = word.size();
        if (position + n > s.size()) return false;
        if (s.compare(position, n, word) != 0) return false;
        position += n;
        return true;
    }

    void append_unescaped_utf8(std::string& out, char first)
    {
        const unsigned char lead = static_cast<unsigned char>(first);
        if (lead < 0x80)
        {
            out.push_back(first);
            return;
        }

        unsigned continuation_count = 0;
        unsigned code_point = 0;
        unsigned minimum = 0;
        if (lead >= 0xC2 && lead <= 0xDF)
        {
            continuation_count = 1;
            code_point = lead & 0x1F;
            minimum = 0x80;
        }
        else if (lead >= 0xE0 && lead <= 0xEF)
        {
            continuation_count = 2;
            code_point = lead & 0x0F;
            minimum = 0x800;
        }
        else if (lead >= 0xF0 && lead <= 0xF4)
        {
            continuation_count = 3;
            code_point = lead & 0x07;
            minimum = 0x10000;
        }
        else
        {
            fail("invalid UTF-8 in string");
        }

        const std::size_t sequence_start = position - 1;
        if (position + continuation_count > s.size()) fail("truncated UTF-8 in string");
        for (unsigned i = 0; i < continuation_count; ++i)
        {
            const unsigned char byte = static_cast<unsigned char>(s[position++]);
            if ((byte & 0xC0) != 0x80) fail("invalid UTF-8 continuation byte in string");
            code_point = (code_point << 6) | (byte & 0x3F);
        }
        if (code_point < minimum || code_point > 0x10FFFF
            || (code_point >= 0xD800 && code_point <= 0xDFFF))
            fail("invalid UTF-8 code point in string");
        out.append(s.substr(sequence_start, continuation_count + 1));
    }

    unsigned read_hex_quad()
    {
        if (position + 4 > s.size()) fail("bad \\u");

        unsigned value = 0;
        for (int i = 0; i < 4; ++i)
        {
            const char digit = s[position++];
            value <<= 4;
            if (digit >= '0' && digit <= '9') value |= unsigned(digit - '0');
            else if (digit >= 'a' && digit <= 'f') value |= unsigned(digit - 'a' + 10);
            else if (digit >= 'A' && digit <= 'F') value |= unsigned(digit - 'A' + 10);
            else fail("bad hex in \\u");
        }
        return value;
    }

    void append_unicode_escape(std::string& out)
    {
        unsigned code = read_hex_quad();
        if (code >= 0xD800 && code <= 0xDBFF)
        {
            if (position + 1 >= s.size() || s[position] != '\\' || s[position + 1] != 'u')
                fail("unpaired high surrogate in \\u escape");
            position += 2;
            const unsigned low = read_hex_quad();
            if (low < 0xDC00 || low > 0xDFFF)
                fail("unpaired high surrogate in \\u escape");
            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
        }
        else if (code >= 0xDC00 && code <= 0xDFFF)
            fail("unpaired low surrogate in \\u escape");

        append_utf8(out, code);
    }

    void append_escape(std::string& out)
    {
        if (position >= s.size()) fail("bad escape");

        switch (s[position++])
        {
        case '"':  out.push_back('"');  break;
        case '\\': out.push_back('\\'); break;
        case '/':  out.push_back('/');  break;
        case 'n':  out.push_back('\n'); break;
        case 'r':  out.push_back('\r'); break;
        case 't':  out.push_back('\t'); break;
        case 'b':  out.push_back('\b'); break;
        case 'f':  out.push_back('\f'); break;
        case 'u':  append_unicode_escape(out); break;
        default: fail("bad escape");
        }
    }

    std::string parse_string()
    {
        if (consume() != '"') fail("expected '\"'");
        std::string out;
        while (position < s.size())
        {
            const char c = s[position++];
            if (c == '"') return out;

            if (c != '\\')
            {
                if (static_cast<unsigned char>(c) < 0x20)
                    fail("unescaped control character in string");
                append_unescaped_utf8(out, c);
                continue;
            }

            append_escape(out);
        }
        fail("unterminated string");
    }

    void consume_digits(const char* error)
    {
        if (position >= s.size() || !std::isdigit(static_cast<unsigned char>(s[position])))
            fail(error);
        while (position < s.size() && std::isdigit(static_cast<unsigned char>(s[position])))
            ++position;
    }

    void consume_integer_part()
    {
        if (position >= s.size()) fail("bad number");
        if (s[position] != '0')
        {
            if (s[position] < '1' || s[position] > '9') fail("bad number");
            consume_digits("bad number");
            return;
        }

        ++position;
        if (position < s.size() && std::isdigit(static_cast<unsigned char>(s[position])))
            fail("leading zero in number");
    }

    Json parse_number()
    {
        skip_ws();
        const std::size_t start = position;
        if (position < s.size() && s[position] == '-') ++position;
        consume_integer_part();
        if (position < s.size() && s[position] == '.')
        {
            ++position;
            consume_digits("fraction requires a digit");
        }
        if (position < s.size() && is_one_of(s[position], 'e', 'E'))
        {
            ++position;
            if (position < s.size() && is_one_of(s[position], '+', '-')) ++position;
            consume_digits("exponent requires a digit");
        }
        double value = 0.0;
        if (!parse_double_exact(s.substr(start, position - start), value)) fail("bad number");
        return Json(value);
    }

    Json parse_value(std::size_t depth = 0)
    {
        const char c = peek();
        if (c == '"') return Json(parse_string());
        if (c == '{') return parse_object(depth);
        if (c == '[') return parse_array(depth);
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number();
        if (match("true"))  return Json(true);
        if (match("false")) return Json(false);
        if (match("null"))  return Json{};
        fail(std::format("unexpected character '{}'", c));
    }

    Json parse_object(std::size_t depth)
    {
        if (depth >= max_nesting_depth) fail("maximum nesting depth exceeded");
        if (consume() != '{') fail("expected '{'");
        Json j = Json::make_object();
        skip_ws();
        if (position < s.size() && s[position] == '}') { ++position; return j; }
        while (true)
        {
            std::string key = parse_string();
            skip_ws();
            if (position >= s.size() || s[position] != ':') fail("expected ':'");
            ++position;
            j.as_object().emplace_back(std::move(key), parse_value(depth + 1));
            skip_ws();
            if (position < s.size() && s[position] == ',') { ++position; continue; }
            if (position < s.size() && s[position] == '}') { ++position; return j; }
            fail("expected ',' or '}'");
        }
    }

    Json parse_array(std::size_t depth)
    {
        if (depth >= max_nesting_depth) fail("maximum nesting depth exceeded");
        if (consume() != '[') fail("expected '['");
        Json j = Json::make_array();
        skip_ws();
        if (position < s.size() && s[position] == ']') { ++position; return j; }
        while (true)
        {
            j.push_back(parse_value(depth + 1));
            skip_ws();
            if (position < s.size() && s[position] == ',') { ++position; continue; }
            if (position < s.size() && s[position] == ']') { ++position; return j; }
            fail("expected ',' or ']'");
        }
    }
};

}

Json Json::parse(std::string_view text)
{
    throw_if(text.size() > max_json_input_bytes,
             "JSON parse: input exceeds the 256 MiB safety limit");
    if (text.starts_with("\xEF\xBB\xBF")) text.remove_prefix(3);

    Parser p(text);
    Json v = p.parse_value();
    p.skip_ws();
    throw_if(p.position != text.size(),
             "JSON parse: trailing data");
    return v;
}
void JsonDocument::load(const std::filesystem::path& path)
{
    std::error_code error;
    const std::uintmax_t byte_count = std::filesystem::file_size(path, error);
    throw_if(!error && byte_count > max_json_input_bytes,
             "JSON file exceeds the 256 MiB safety limit: {}", path.string());
    root = Json::parse(read_text_file(path));
}

void JsonDocument::save(const std::filesystem::path& path, int indent) const
{
    std::ofstream out(path);
    throw_if(!out.is_open(),
             "Cannot open JSON file: {}", path.string());
    out << root.dump(indent);
}

void save_json_file(const std::filesystem::path& file_name, const JsonWriter& writer)
{
    std::ofstream file(file_name);

    throw_if(!file.is_open(), "Cannot open file: {}", file_name.string());

    file << writer.c_str();
    file.close();
    throw_if(!file, "Cannot write file: {}", file_name.string());
}

JsonDocument JsonDocument::wrap(std::string_view tag, Json value)
{
    JsonDocument doc;
    doc.set_root(Json::make_object());
    doc.get_root().set(tag, std::move(value));
    return doc;
}
void JsonWriter::open_element(std::string_view name)
{
    Json* parent = stack.empty() ? &root : stack.back();
    if (parent == &root && root.is_null()) root = Json::make_object();

    Json child = Json::make_object();

    if (parent->is_object())
    {
        Json::Object& object = parent->as_object();
        object.emplace_back(std::string(name), std::move(child));
        stack.push_back(&object.back().second);
    }
    else if (parent->is_array())
    {
        Json::Array& array = parent->as_array();
        array.push_back(std::move(child));
        stack.push_back(&array.back());
    }
    else
    {
        throw std::runtime_error("JsonWriter: cannot open_element on non-container");
    }
}

void JsonWriter::begin_array(std::string_view name)
{
    Json* parent = stack.empty() ? &root : stack.back();
    if (parent->is_null()) *parent = Json::make_object();
    throw_if(!parent->is_object(),
             "JsonWriter::begin_array: parent is not an object");
    Json::Object& object = parent->as_object();
    object.emplace_back(std::string(name), Json::make_array());
    stack.push_back(&object.back().second);
}

void JsonWriter::begin_array_object()
{
    throw_if(stack.empty() || !stack.back()->is_array(),
             "JsonWriter::begin_array_object: not in array");
    Json* parent = stack.back();
    Json::Array& array = parent->as_array();
    array.push_back(Json::make_object());
    stack.push_back(&array.back());
}

void JsonWriter::pop_scope()
{
    if (stack.empty()) return;
    stack.pop_back();
}

void JsonWriter::add_field(std::string_view name, Json value)
{
    Json* parent = stack.empty() ? &root : stack.back();
    if (parent->is_null()) *parent = Json::make_object();
    throw_if(!parent->is_object(),
             "JsonWriter::add_field on non-object");
    parent->set(name, std::move(value));
}

void write_json(JsonWriter& writer,
                std::initializer_list<std::pair<const char*, Json>> props)
{
    for (const auto& [key, value] : props)
        writer.add_field(key, value);
}

float read_json_float(const Json* root, std::string_view field)
{
    if (!root) return 0.0f;
    const Json* const v = root->find(field);
    return v ? float(v->as_double()) : 0.0f;
}

long long read_json_index(const Json* root, std::string_view field)
{
    if (!root) return 0;
    const Json* const v = root->find(field);
    return v ? v->as_long() : 0;
}

bool read_json_bool(const Json* root, std::string_view field)
{
    if (!root) return false;
    const Json* const v = root->find(field);
    return v && v->as_bool();
}

std::string read_json_string(const Json* root, std::string_view field)
{
    if (!root) return "";
    const Json* const v = root->find(field);
    return v ? v->as_string() : std::string();
}

float read_json_float(const Json* root, std::string_view field, float fallback)
{
    const Json* const value = root ? root->find(field) : nullptr;
    return value ? float(value->as_double()) : fallback;
}

long long read_json_index(const Json* root, std::string_view field, long long fallback)
{
    const Json* const value = root ? root->find(field) : nullptr;
    return value ? value->as_long() : fallback;
}

bool read_json_bool(const Json* root, std::string_view field, bool fallback)
{
    const Json* const value = root ? root->find(field) : nullptr;
    return value ? value->as_bool() : fallback;
}

std::string read_json_string(const Json* root, std::string_view field, std::string_view fallback)
{
    const Json* const value = root ? root->find(field) : nullptr;
    return value ? value->as_string() : std::string(fallback);
}

std::vector<std::string> read_json_strings(const Json* root, std::string_view field)
{
    const Json* const value = root ? root->find(field) : nullptr;
    if (!value) return {};
    if (!value->is_array()) return get_tokens(value->as_string(), "\n");

    const Json::Array& array = value->as_array();
    std::vector<std::string> values(array.size());
    std::ranges::transform(array, values.begin(),
                      [](const Json& item) { return item.as_string(); });
    return values;
}

std::string read_json_string_fallback(const Json* root,
                                      std::initializer_list<std::string_view> names)
{
    if (!root) return "";
    for (const auto& name : names)
    {
        const Json* const v = root->find(name);
        if (v) return v->as_string();
    }
    return "";
}

const Json* require_json_field(const Json* root, std::string_view field)
{
    throw_if(!root, "JSON: missing root for field '{}'", field);
    const Json* const v = root->find(field);
    throw_if(!v, "JSON: missing required field '{}'", field);
    return v;
}

JsonDocument load_json_file(const std::filesystem::path& file_name)
{
    JsonDocument doc;
    doc.load(file_name);
    return doc;
}

const Json* get_json_root(const JsonDocument& document, std::string_view tag)
{
    const Json* const v = document.first_child(tag);
    throw_if(!v, "JSON: missing root tag '{}'", tag);
    return v;
}

}
