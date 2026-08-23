//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   J S O N   M I N I M A L   S U P P O R T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

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

namespace opennn
{

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

bool Json::has(std::string_view key) const
{
    return find(key) != nullptr;
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
    case Number: return (long long)(std::get<double>(value));
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
        const char* const first = string.data();
        const char* const last = first + string.size();
        const auto [end, error] = std::from_chars(first, last, number);
        throw_if(error != std::errc{} || end != last,
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

        // Before the cast, not after: converting a NaN, an infinity or anything
        // past 2^63 to long long is undefined, and to_chars would then write
        // "nan"/"inf", which this parser rejects on the way back in.
        if (!std::isfinite(number)) { out += "null"; return; }

        // The integrality test is a trunc comparison rather than a round trip
        // through long long, so the cast happens only where 1e15 has already
        // proved it is in range.
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

struct Parser
{
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
                out.push_back(c);
                continue;
            }

            if (position >= s.size()) fail("bad escape");

            const char e = s[position++];
            switch (e)
            {
            case '"':  out.push_back('"');  break;
            case '\\': out.push_back('\\'); break;
            case '/':  out.push_back('/');  break;
            case 'n':  out.push_back('\n'); break;
            case 'r':  out.push_back('\r'); break;
            case 't':  out.push_back('\t'); break;
            case 'b':  out.push_back('\b'); break;
            case 'f':  out.push_back('\f'); break;
            case 'u':
            {
                const auto read_four_hex = [&]() -> unsigned
                {
                    if (position + 4 > s.size()) fail("bad \\u");

                    unsigned value = 0;
                    for (int i = 0; i < 4; ++i)
                    {
                        const char h = s[position++];
                        value <<= 4;
                        if (h >= '0' && h <= '9')      value |= unsigned(h - '0');
                        else if (h >= 'a' && h <= 'f') value |= unsigned(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') value |= unsigned(h - 'A' + 10);
                        else fail("bad hex in \\u");
                    }
                    return value;
                };

                unsigned code = read_four_hex();

                // A non-BMP character is written as a surrogate pair, which is
                // how json.dumps escapes every emoji in a vocabulary file. Each
                // half encoded on its own produced six bytes of CESU-8 that no
                // tokenizer could match against the real four-byte character.
                if (code >= 0xD800 && code <= 0xDBFF
                    && position + 1 < s.size() && s[position] == '\\' && s[position + 1] == 'u')
                {
                    const size_t saved_position = position;
                    position += 2;
                    const unsigned low = read_four_hex();

                    if (low >= 0xDC00 && low <= 0xDFFF)
                        code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                    else
                        position = saved_position;
                }

                append_utf8(out, code);
                break;
            }
            default: fail("bad escape");
            }
        }
        fail("unterminated string");
    }

    Json parse_number()
    {
        skip_ws();
        const std::size_t start = position;
        if (position < s.size() && s[position] == '-') ++position;
        while (position < s.size() && std::isdigit(static_cast<unsigned char>(s[position]))) ++position;
        if (position < s.size() && s[position] == '.') { ++position; while (position < s.size() && std::isdigit(static_cast<unsigned char>(s[position]))) ++position; }
        if (position < s.size() && is_one_of(s[position], 'e', 'E'))
        {
            ++position;
            if (position < s.size() && is_one_of(s[position], '+', '-')) ++position;
            while (position < s.size() && std::isdigit(static_cast<unsigned char>(s[position]))) ++position;
        }
        double value = 0.0;
        const char* const first = s.data() + start;
        const char* const last = s.data() + position;
        const auto [ptr, ec] = std::from_chars(first, last, value);
        if (ec != std::errc() || ptr != last) fail("bad number");
        return Json(value);
    }

    Json parse_value()
    {
        const char c = peek();
        if (c == '"') return Json(parse_string());
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number();
        if (match("true"))  return Json(true);
        if (match("false")) return Json(false);
        if (match("null"))  return Json{};
        fail(std::format("unexpected character '{}'", c));
    }

    Json parse_object()
    {
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
            j.as_object().emplace_back(std::move(key), parse_value());
            skip_ws();
            if (position < s.size() && s[position] == ',') { ++position; continue; }
            if (position < s.size() && s[position] == '}') { ++position; return j; }
            fail("expected ',' or '}'");
        }
    }

    Json parse_array()
    {
        if (consume() != '[') fail("expected '['");
        Json j = Json::make_array();
        skip_ws();
        if (position < s.size() && s[position] == ']') { ++position; return j; }
        while (true)
        {
            j.push_back(parse_value());
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
    // RFC 8259 lets a parser ignore a leading byte-order mark, and PowerShell
    // writes one on every redirect, so a model file merely touched by a Windows
    // tool stopped loading with "unexpected character" at position 0.
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
