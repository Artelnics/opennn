//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   J S O N   M I N I M A L   S U P P O R T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "json.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

namespace opennn
{

Json Json::make_object() { Json j; j.kind = Kind::Object; return j; }
Json Json::make_array () { Json j; j.kind = Kind::Array;  return j; }

bool Json::has(const string& key) const
{
    return find(key) != nullptr;
}

const Json* Json::find(const string& key) const
{
    if (!is_object()) return nullptr;
    for (const auto& kv : object_value)
        if (kv.first == key) return &kv.second;
    return nullptr;
}

const Json& Json::at(const string& key) const
{
    const Json* v = find(key);
    if (!v) throw runtime_error("JSON: missing key '" + key + "'");
    return *v;
}

Json& Json::operator[](const string& key)
{
    if (!is_object()) { kind = Kind::Object; object_value.clear(); }
    for (auto& kv : object_value)
        if (kv.first == key) return kv.second;
    object_value.emplace_back(key, Json{});
    return object_value.back().second;
}

Json& Json::set(const string& key, Json value)
{
    (*this)[key] = move(value);
    return *this;
}

void Json::push_back(Json value)
{
    if (!is_array()) { kind = Kind::Array; array_value.clear(); }
    array_value.push_back(move(value));
}

string Json::as_string() const
{
    using enum Kind;
    switch (kind)
    {
    case Null:   return "";
    case Bool:   return bool_value ? "1" : "0";
    case Number: {
        ostringstream s; s.precision(10); s << number_value; return s.str();
    }
    case String: return string_value;
    case Array:
    case Object: return dump(0);
    }

    return "";
}

long Json::as_long() const
{
    using enum Kind;
    switch (kind)
    {
    case Number: return long(number_value);
    case Bool:   return bool_value ? 1 : 0;
    case String: return string_value.empty() ? 0L : std::stol(string_value);
    case Null:
    case Array:
    case Object: return 0;
    }

    return 0;
}

double Json::as_double() const
{
    using enum Kind;
    switch (kind)
    {
    case Number: return number_value;
    case Bool:   return bool_value ? 1.0 : 0.0;
    case String: return string_value.empty() ? 0.0 : std::stod(string_value);
    case Null:
    case Array:
    case Object: return 0.0;
    }

    return 0.0;
}

bool Json::as_bool() const
{
    using enum Kind;
    switch (kind)
    {
    case Bool:   return bool_value;
    case Number: return number_value != 0.0;
    case String: return string_value == "1" || string_value == "true";
    case Null:
    case Array:
    case Object: return false;
    }

    return false;
}
static void escape_string(string& out, const string& s)
{
    out.push_back('"');
    for (char c : s)
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
                snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                out += buf;
            }
            else out.push_back(c);
        }
    }
    out.push_back('"');
}

static void dump_value(string& out, const Json& v, int indent, int depth);

static void dump_indent(string& out, int indent, int depth)
{
    if (indent <= 0) return;
    out.push_back('\n');
    for (int i = 0; i < indent * depth; ++i) out.push_back(' ');
}

static void dump_value(string& out, const Json& v, int indent, int depth)
{
    using enum Json::Kind;
    switch (v.kind)
    {
    case Null:   out += "null"; return;
    case Bool:   out += (v.bool_value ? "true" : "false"); return;
    case Number: {
        char buf[32];
        const long long as_int = static_cast<long long>(v.number_value);
        if (v.number_value == static_cast<double>(as_int) && abs(v.number_value) < 1e15)
            snprintf(buf, sizeof(buf), "%lld", as_int);
        else
            snprintf(buf, sizeof(buf), "%.10g", v.number_value);
        out += buf;
        return;
    }
    case String: escape_string(out, v.string_value); return;
    case Array:
        if (v.array_value.empty()) { out += "[]"; return; }
        out.push_back('[');
        for (size_t i = 0; i < v.array_value.size(); ++i)
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
        for (size_t i = 0; i < v.object_value.size(); ++i)
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

string Json::dump(int indent) const
{
    string out;
    dump_value(out, *this, indent, 0);
    return out;
}

// -------- Parser --------

namespace {

struct Parser
{
    const string& s;
    size_t i = 0;

    explicit Parser(const string& text) : s(text) {}

    void skip_ws()
    {
        while (i < s.size())
        {
            char c = s[i];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++i;
            else break;
        }
    }

    [[noreturn]] void fail(const string& msg) const
    {
        throw runtime_error("JSON parse error at " + to_string(i) + ": " + msg);
    }

    char peek()
    {
        skip_ws();
        if (i >= s.size()) fail("unexpected end of input");
        return s[i];
    }

    char consume()
    {
        skip_ws();
        if (i >= s.size()) fail("unexpected end of input");
        return s[i++];
    }

    bool match(const char* word)
    {
        skip_ws();
        const size_t n = strlen(word);
        if (i + n > s.size()) return false;
        if (s.compare(i, n, word) != 0) return false;
        i += n;
        return true;
    }

    string parse_string()
    {
        if (consume() != '"') fail("expected '\"'");
        string out;
        while (i < s.size())
        {
            char c = s[i++];
            if (c == '"') return out;
            if (c == '\\')
            {
                if (i >= s.size()) fail("bad escape");
                char e = s[i++];
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
                case 'u': {
                    if (i + 4 > s.size()) fail("bad \\u");
                    unsigned code = 0;
                    for (int k = 0; k < 4; ++k)
                    {
                        char h = s[i++];
                        code <<= 4;
                        if (h >= '0' && h <= '9')      code |= unsigned(h - '0');
                        else if (h >= 'a' && h <= 'f') code |= unsigned(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') code |= unsigned(h - 'A' + 10);
                        else fail("bad hex in \\u");
                    }
                    // Minimal UTF-8 encoding (BMP only).
                    if (code < 0x80) out.push_back(char(code));
                    else if (code < 0x800)
                    {
                        out.push_back(char(0xC0 | (code >> 6)));
                        out.push_back(char(0x80 | (code & 0x3F)));
                    }
                    else
                    {
                        out.push_back(char(0xE0 | (code >> 12)));
                        out.push_back(char(0x80 | ((code >> 6) & 0x3F)));
                        out.push_back(char(0x80 | (code & 0x3F)));
                    }
                    break;
                }
                default: fail("bad escape");
                }
            }
            else out.push_back(c);
        }
        fail("unterminated string");
    }

    Json parse_number()
    {
        skip_ws();
        const size_t start = i;
        if (i < s.size() && s[i] == '-') ++i;
        while (i < s.size() && isdigit(static_cast<unsigned char>(s[i]))) ++i;
        if (i < s.size() && s[i] == '.') { ++i; while (i < s.size() && isdigit(static_cast<unsigned char>(s[i]))) ++i; }
        if (i < s.size() && (s[i] == 'e' || s[i] == 'E'))
        {
            ++i;
            if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
            while (i < s.size() && isdigit(static_cast<unsigned char>(s[i]))) ++i;
        }
        Json j;
        j.kind = Json::Kind::Number;
        j.number_value = stod(s.substr(start, i - start));
        return j;
    }

    Json parse_value()
    {
        char c = peek();
        if (c == '"') { Json j; j.kind = Json::Kind::String; j.string_value = parse_string(); return j; }
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '-' || isdigit(static_cast<unsigned char>(c))) return parse_number();
        if (match("true"))  { Json j; j.kind = Json::Kind::Bool; j.bool_value = true;  return j; }
        if (match("false")) { Json j; j.kind = Json::Kind::Bool; j.bool_value = false; return j; }
        if (match("null"))  return Json{};
        fail(string("unexpected character '") + c + "'");
    }

    Json parse_object()
    {
        if (consume() != '{') fail("expected '{'");
        Json j = Json::make_object();
        skip_ws();
        if (i < s.size() && s[i] == '}') { ++i; return j; }
        while (true)
        {
            string key = parse_string();
            skip_ws();
            if (i >= s.size() || s[i] != ':') fail("expected ':'");
            ++i;
            j.object_value.emplace_back(move(key), parse_value());
            skip_ws();
            if (i < s.size() && s[i] == ',') { ++i; continue; }
            if (i < s.size() && s[i] == '}') { ++i; return j; }
            fail("expected ',' or '}'");
        }
    }

    Json parse_array()
    {
        if (consume() != '[') fail("expected '['");
        Json j = Json::make_array();
        skip_ws();
        if (i < s.size() && s[i] == ']') { ++i; return j; }
        while (true)
        {
            j.array_value.push_back(parse_value());
            skip_ws();
            if (i < s.size() && s[i] == ',') { ++i; continue; }
            if (i < s.size() && s[i] == ']') { ++i; return j; }
            fail("expected ',' or ']'");
        }
    }
};

}

Json Json::parse(const string& text)
{
    Parser p(text);
    Json v = p.parse_value();
    p.skip_ws();
    if (p.i != text.size())
        throw runtime_error("JSON parse: trailing data");
    return v;
}
void JsonDocument::load(const filesystem::path& path)
{
    ifstream in(path);
    if (!in.is_open())
        throw runtime_error("Cannot open JSON file: " + path.string());
    stringstream ss;
    ss << in.rdbuf();
    root = Json::parse(ss.str());
}

void JsonDocument::save(const filesystem::path& path, int indent) const
{
    ofstream out(path);
    if (!out.is_open())
        throw runtime_error("Cannot open JSON file: " + path.string());
    out << root.dump(indent);
}

const Json* JsonDocument::first_child(const string& name) const
{
    return root.find(name);
}

JsonDocument JsonDocument::wrap(const string& tag, Json value)
{
    JsonDocument doc;
    doc.root = Json::make_object();
    doc.root.set(tag, move(value));
    return doc;
}
void JsonWriter::open_element(const string& name)
{
    Json* parent = stack.empty() ? &root : stack.back();
    if (parent == &root && root.kind == Json::Kind::Null) root = Json::make_object();

    Json child = Json::make_object();

    if (parent->is_object())
    {
        parent->object_value.emplace_back(name, move(child));
        stack.push_back(&parent->object_value.back().second);
    }
    else if (parent->is_array())
    {
        parent->array_value.push_back(move(child));
        stack.push_back(&parent->array_value.back());
    }
    else
    {
        throw runtime_error("JsonWriter: cannot open_element on non-container");
    }
    name_stack.push_back(name);
}

void JsonWriter::close_element()
{
    if (stack.empty()) return;
    stack.pop_back();
    if (!name_stack.empty()) name_stack.pop_back();
}

void JsonWriter::begin_array(const string& name)
{
    Json* parent = stack.empty() ? &root : stack.back();
    if (parent->kind == Json::Kind::Null) *parent = Json::make_object();
    if (!parent->is_object())
        throw runtime_error("JsonWriter::begin_array: parent is not an object");
    parent->object_value.emplace_back(name, Json::make_array());
    stack.push_back(&parent->object_value.back().second);
    name_stack.push_back(name);
}

void JsonWriter::end_array()
{
    if (stack.empty()) return;
    stack.pop_back();
    if (!name_stack.empty()) name_stack.pop_back();
}

void JsonWriter::begin_array_object()
{
    if (stack.empty() || !stack.back()->is_array())
        throw runtime_error("JsonWriter::begin_array_object: not in array");
    Json* parent = stack.back();
    parent->array_value.push_back(Json::make_object());
    stack.push_back(&parent->array_value.back());
    name_stack.push_back("");
}

void JsonWriter::end_array_object()
{
    if (stack.empty()) return;
    stack.pop_back();
    if (!name_stack.empty()) name_stack.pop_back();
}

void JsonWriter::add_field(const string& name, const string& value)
{
    Json* parent = stack.empty() ? &root : stack.back();
    if (parent->kind == Json::Kind::Null) *parent = Json::make_object();
    if (!parent->is_object())
        throw runtime_error("JsonWriter::add_field on non-object");
    parent->set(name, Json(value));
}

string JsonWriter::c_str(int indent) const
{
    return root.dump(indent);
}
void add_json_field(JsonWriter& writer,
                    const string& name,
                    const string& value)
{
    writer.add_field(name, value);
}

void write_json(JsonWriter& writer,
                initializer_list<pair<const char*, string>> props)
{
    for (const auto& kv : props)
        writer.add_field(kv.first, kv.second);
}

float read_json_type(const Json* root, const string& field)
{
    if (!root) return 0.0f;
    const Json* v = root->find(field);
    return v ? float(v->as_double()) : 0.0f;
}

long read_json_index(const Json* root, const string& field)
{
    if (!root) return 0;
    const Json* v = root->find(field);
    return v ? v->as_long() : 0;
}

bool read_json_bool(const Json* root, const string& field)
{
    if (!root) return false;
    const Json* v = root->find(field);
    return v ? v->as_bool() : false;
}

string read_json_string(const Json* root, const string& field)
{
    if (!root) return "";
    const Json* v = root->find(field);
    return v ? v->as_string() : string();
}

string read_json_string_fallback(const Json* root,
                                      initializer_list<string> names)
{
    if (!root) return "";
    for (const auto& name : names)
    {
        const Json* v = root->find(name);
        if (v) return v->as_string();
    }
    return "";
}

const Json* require_json_field(const Json* root, const string& field)
{
    if (!root) throw runtime_error("JSON: missing root for field '" + field + "'");
    const Json* v = root->find(field);
    if (!v) throw runtime_error("JSON: missing required field '" + field + "'");
    return v;
}

JsonDocument load_json_file(const filesystem::path& file_name)
{
    JsonDocument doc;
    doc.load(file_name);
    return doc;
}

const Json* get_json_root(const JsonDocument& document, const string& tag)
{
    const Json* v = document.first_child(tag);
    if (!v) throw runtime_error("JSON: missing root tag '" + tag + "'");
    return v;
}

}
