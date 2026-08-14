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

Json Json::make_object() { Json j; j.kind = Kind::Object; return j; }
Json Json::make_array () { Json j; j.kind = Kind::Array;  return j; }

bool Json::has(std::string_view key) const
{
    return find(key) != nullptr;
}

const Json* Json::find(std::string_view key) const
{
    if (!is_object()) return nullptr;
    const auto it = std::ranges::find_if(object_value,
                                         [key](const auto& item) { return item.first == key; });
    return it != object_value.end() ? &it->second : nullptr;
}

const Json& Json::at(std::string_view key) const
{
    const Json* const v = find(key);
    throw_if(!v, "JSON: missing key '{}'", key);
    return *v;
}

Json& Json::operator[](std::string_view key)
{
    if (!is_object()) { kind = Kind::Object; object_value.clear(); }
    for (auto& [k, v] : object_value)
        if (k == key) return v;
    object_value.emplace_back(std::string(key), Json{});
    return object_value.back().second;
}

Json& Json::set(std::string_view key, Json value)
{
    (*this)[key] = std::move(value);
    return *this;
}

void Json::push_back(Json value)
{
    if (!is_array()) { kind = Kind::Array; array_value.clear(); }
    array_value.push_back(std::move(value));
}

std::string Json::as_string() const
{
    using enum Kind;
    switch (kind)
    {
    case Null:   return "";
    case Bool:   return bool_value ? "1" : "0";
    case Number: return std::format("{:.10g}", number_value);
    case String: return string_value;
    case Array:
    case Object: return dump(0);
    }

    throw std::runtime_error("JSON: invalid value kind");
}

long long Json::as_long() const
{
    using enum Kind;
    switch (kind)
    {
    case Number: return (long long)(number_value);
    case Bool:   return bool_value ? 1 : 0;
    case String:
        if (string_value.empty()) return 0LL;
        return parse_number<long long>(string_value, "JSON", "integer");
    case Null:
    case Array:
    case Object: return 0;
    }

    throw std::runtime_error("JSON: invalid value kind");
}

double Json::as_double() const
{
    using enum Kind;
    switch (kind)
    {
    case Number: return number_value;
    case Bool:   return bool_value ? 1.0 : 0.0;
    case String: {
        if (string_value.empty()) return 0.0;
        double value = 0.0;
        const char* const first = string_value.data();
        const char* const last = first + string_value.size();
        const auto [end, error] = std::from_chars(first, last, value);
        throw_if(error != std::errc{} || end != last,
                 "JSON: invalid numeric value '{}'", string_value);
        return value;
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
    switch (kind)
    {
    case Bool:   return bool_value;
    case Number: return number_value != 0.0;
    case String: return contains({"1", "true"}, string_value);
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
    switch (v.kind)
    {
    case Null:   out += "null"; return;
    case Bool:   out += (v.bool_value ? "true" : "false"); return;
    case Number: {
        char buf[32];
        const long long as_int = static_cast<long long>(v.number_value);
        if (v.number_value == static_cast<double>(as_int) && std::abs(v.number_value) < 1e15)
            std::snprintf(buf, sizeof(buf), "%lld", as_int);
        else
        {
            auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf) - 1, v.number_value);
            *ptr = '\0';
        }
        out += buf;
        return;
    }
    case String: escape_string(out, v.string_value); return;
    case Array:
        if (v.array_value.empty()) { out += "[]"; return; }
        out.push_back('[');
        for (std::size_t i = 0; i < v.array_value.size(); ++i)
        {
            dump_indent(out, indent, depth + 1);
            dump_value(out, v.array_value[i], indent, depth + 1);
            if (i + 1 < v.array_value.size()) out.push_back(',');
        }
        dump_indent(out, indent, depth);
        out.push_back(']');
        return;
    case Object:
        if (v.object_value.empty()) { out += "{}"; return; }
        out.push_back('{');
        for (std::size_t i = 0; i < v.object_value.size(); ++i)
        {
            dump_indent(out, indent, depth + 1);
            escape_string(out, v.object_value[i].first);
            out += indent > 0 ? ": " : ":";
            dump_value(out, v.object_value[i].second, indent, depth + 1);
            if (i + 1 < v.object_value.size()) out.push_back(',');
        }
        dump_indent(out, indent, depth);
        out.push_back('}');
        return;
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

    bool match(const char* word)
    {
        skip_ws();
        const std::size_t n = std::strlen(word);
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
                if (position + 4 > s.size()) fail("bad \\u");

                unsigned code = 0;
                for (int i = 0; i < 4; ++i)
                {
                    const char h = s[position++];
                    code <<= 4;
                    if (h >= '0' && h <= '9')      code |= unsigned(h - '0');
                    else if (h >= 'a' && h <= 'f') code |= unsigned(h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') code |= unsigned(h - 'A' + 10);
                    else fail("bad hex in \\u");
                }

                if (code < 0x80)
                {
                    out.push_back(char(code));
                    break;
                }

                if (code < 0x800)
                {
                    out.push_back(char(0xC0 | (code >> 6)));
                    out.push_back(char(0x80 | (code & 0x3F)));
                    break;
                }

                out.push_back(char(0xE0 | (code >> 12)));
                out.push_back(char(0x80 | ((code >> 6) & 0x3F)));
                out.push_back(char(0x80 | (code & 0x3F)));
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
        Json j;
        j.kind = Json::Kind::Number;
        const char* const first = s.data() + start;
        const char* const last = s.data() + position;
        const auto [ptr, ec] = std::from_chars(first, last, j.number_value);
        if (ec != std::errc() || ptr != last) fail("bad number");
        return j;
    }

    Json parse_value()
    {
        const char c = peek();
        if (c == '"') { Json j; j.kind = Json::Kind::String; j.string_value = parse_string(); return j; }
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number();
        if (match("true"))  { Json j; j.kind = Json::Kind::Bool; j.bool_value = true;  return j; }
        if (match("false")) { Json j; j.kind = Json::Kind::Bool; j.bool_value = false; return j; }
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
            j.object_value.emplace_back(std::move(key), parse_value());
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
            j.array_value.push_back(parse_value());
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
}

const Json* JsonDocument::first_child(std::string_view name) const
{
    return root.find(name);
}

JsonDocument JsonDocument::wrap(std::string_view tag, Json value)
{
    JsonDocument doc;
    doc.root = Json::make_object();
    doc.root.set(tag, std::move(value));
    return doc;
}
void JsonWriter::open_element(std::string_view name)
{
    Json* parent = stack.empty() ? &root : stack.back();
    if (parent == &root && root.kind == Json::Kind::Null) root = Json::make_object();

    Json child = Json::make_object();

    if (parent->is_object())
    {
        parent->object_value.emplace_back(std::string(name), std::move(child));
        stack.push_back(&parent->object_value.back().second);
    }
    else if (parent->is_array())
    {
        parent->array_value.push_back(std::move(child));
        stack.push_back(&parent->array_value.back());
    }
    else
    {
        throw std::runtime_error("JsonWriter: cannot open_element on non-container");
    }
}

void JsonWriter::close_element()
{
    pop_scope();
}

void JsonWriter::begin_array(std::string_view name)
{
    Json* parent = stack.empty() ? &root : stack.back();
    if (parent->kind == Json::Kind::Null) *parent = Json::make_object();
    throw_if(!parent->is_object(),
             "JsonWriter::begin_array: parent is not an object");
    parent->object_value.emplace_back(std::string(name), Json::make_array());
    stack.push_back(&parent->object_value.back().second);
}

void JsonWriter::end_array()
{
    pop_scope();
}

void JsonWriter::begin_array_object()
{
    throw_if(stack.empty() || !stack.back()->is_array(),
             "JsonWriter::begin_array_object: not in array");
    Json* parent = stack.back();
    parent->array_value.push_back(Json::make_object());
    stack.push_back(&parent->array_value.back());
}

void JsonWriter::end_array_object()
{
    pop_scope();
}

void JsonWriter::pop_scope()
{
    if (stack.empty()) return;
    stack.pop_back();
}

void JsonWriter::add_field(std::string_view name, Json value)
{
    Json* parent = stack.empty() ? &root : stack.back();
    if (parent->kind == Json::Kind::Null) *parent = Json::make_object();
    throw_if(!parent->is_object(),
             "JsonWriter::add_field on non-object");
    parent->set(name, std::move(value));
}

std::string JsonWriter::c_str(int indent) const
{
    return root.dump(indent);
}
void add_json_field(JsonWriter& writer,
                    std::string_view name,
                    Json value)
{
    writer.add_field(name, std::move(value));
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

std::vector<std::string> read_json_strings(const Json* root, std::string_view field)
{
    const Json* const value = root ? root->find(field) : nullptr;
    if (!value) return {};
    if (!value->is_array()) return get_tokens(value->as_string(), "\n");

    std::vector<std::string> values(value->array_value.size());
    std::ranges::transform(value->array_value, values.begin(),
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
