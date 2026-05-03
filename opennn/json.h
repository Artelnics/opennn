//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   J S O N   M I N I M A L   S U P P O R T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <filesystem>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace opennn
{

class Json
{
public:
    enum class Kind { Null, Bool, Number, String, Array, Object };

    Kind                                 kind = Kind::Null;
    bool                                 bool_value = false;
    double                               number_value = 0.0;
    std::string                          string_value;
    std::vector<Json>                    array_value;
    std::vector<std::pair<std::string, Json>> object_value;

    Json() = default;
    Json(bool b)                : kind(Kind::Bool),   bool_value(b)   {}
    Json(int i)                 : kind(Kind::Number), number_value(i) {}
    Json(long i)                : kind(Kind::Number), number_value(double(i)) {}
    Json(long long i)           : kind(Kind::Number), number_value(double(i)) {}
    Json(unsigned int i)        : kind(Kind::Number), number_value(double(i)) {}
    Json(unsigned long i)       : kind(Kind::Number), number_value(double(i)) {}
    Json(unsigned long long i)  : kind(Kind::Number), number_value(double(i)) {}
    Json(double d)              : kind(Kind::Number), number_value(d) {}
    Json(float d)               : kind(Kind::Number), number_value(double(d)) {}
    Json(const char* s)         : kind(Kind::String), string_value(s) {}
    Json(const std::string& s)  : kind(Kind::String), string_value(s) {}

    static Json make_object();
    static Json make_array();

    bool is_null()   const { return kind == Kind::Null; }
    bool is_bool()   const { return kind == Kind::Bool; }
    bool is_number() const { return kind == Kind::Number; }
    bool is_string() const { return kind == Kind::String; }
    bool is_array()  const { return kind == Kind::Array; }
    bool is_object() const { return kind == Kind::Object; }

    // Object accessors (for objects only).
    bool         has(const std::string& key) const;
    const Json*  find(const std::string& key) const;
    const Json*  first_child(const std::string& key) const { return find(key); }
    const Json&  at(const std::string& key) const;
    Json&        operator[](const std::string& key);

    // Append to object (preserves insertion order).
    Json& set(const std::string& key, Json value);

    // Append to array.
    void push_back(Json value);

    // Typed accessors with conversion (parses strings if needed).
    std::string as_string() const;
    long        as_long()   const;
    double      as_double() const;
    bool        as_bool()   const;

    // Parse / serialize.
    static Json  parse(const std::string& text);
    std::string  dump(int indent = 2) const;
};

class JsonDocument
{
public:
    Json root;

    void load(const std::filesystem::path& path);
    void save(const std::filesystem::path& path, int indent = 2) const;

    // Returns pointer to a top-level field or nullptr.
    const Json* first_child(const std::string& name) const;

    // Zero-arg form: returns the root itself (used when callers wrap a single
    // object as a document and want a pointer-form accessor for read_json_*).
    const Json* first_child() const { return &root; }

    // Build a fresh document whose root is an object {tag: value}.
    static JsonDocument wrap(const std::string& tag, Json value);
};

// Incremental writer (mirrors the old XmlPrinter API). Only one root object.
class JsonWriter
{
public:
    void open_element(const std::string& name);
    void close_element();

    void begin_array(const std::string& name);
    void end_array();
    void begin_array_object();
    void end_array_object();

    // Inner element of named-string form (matches XML <Name>value</Name>).
    void add_field(const std::string& name, const std::string& value);

    std::string c_str(int indent = 2) const;

private:
    Json                root;
    std::vector<Json*>  stack;  // path of containers currently open
    std::vector<std::string> name_stack; // for opened named scalar/object
};

// Helpers — writing.

void add_json_field(JsonWriter& writer,
                    const std::string& name,
                    const std::string& value);

void write_json(JsonWriter& writer,
                std::initializer_list<std::pair<const char*, std::string>> props);

// Helpers — reading.

float       read_json_type   (const Json* root, const std::string& field);
long        read_json_index  (const Json* root, const std::string& field);
bool        read_json_bool   (const Json* root, const std::string& field);
std::string read_json_string (const Json* root, const std::string& field);

std::string read_json_string_fallback(const Json* root,
                                      std::initializer_list<std::string> names);

const Json* require_json_field(const Json* root, const std::string& field);

template<typename Func>
void for_json_items(const Json* parent, const char* tag, long count, Func func)
{
    if (!parent || !parent->is_object())
        throw std::runtime_error(std::string("Missing JSON parent for: ") + tag);

    const Json* arr = parent->find(tag);
    if (!arr || !arr->is_array() || long(arr->array_value.size()) != count)
        throw std::runtime_error(std::string("Missing or wrong-size JSON array: ") + tag);

    for (long i = 0; i < count; i++)
        func(i, &arr->array_value[size_t(i)]);
}

JsonDocument load_json_file(const std::filesystem::path& file_name);
const Json*  get_json_root (const JsonDocument& document, const std::string& tag);

}
