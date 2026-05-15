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
    string                          string_value;
    vector<Json>                    array_value;
    vector<pair<string, Json>> object_value;

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
    Json(const string& s)  : kind(Kind::String), string_value(s) {}

    static Json make_object();
    static Json make_array();

    bool is_null()   const { return kind == Kind::Null; }
    bool is_bool()   const { return kind == Kind::Bool; }
    bool is_number() const { return kind == Kind::Number; }
    bool is_string() const { return kind == Kind::String; }
    bool is_array()  const { return kind == Kind::Array; }
    bool is_object() const { return kind == Kind::Object; }
    bool         has(const string& key) const;
    const Json*  find(const string& key) const;
    const Json*  first_child(const string& key) const { return find(key); }
    const Json&  at(const string& key) const;
    Json&        operator[](const string& key);
    Json& set(const string& key, Json value);
    void push_back(Json value);
    string as_string() const;
    long        as_long()   const;
    double      as_double() const;
    bool        as_bool()   const;
    static Json  parse(const string& text);
    string  dump(int indent = 2) const;
};

class JsonDocument
{
public:
    Json root;

    void load(const filesystem::path& path);
    void save(const filesystem::path& path, int indent = 2) const;
    const Json* first_child(const string& name) const;
    const Json* first_child() const { return &root; }
    static JsonDocument wrap(const string& tag, Json value);
};

// Incremental writer (mirrors the old XmlPrinter API). Only one root object.
class JsonWriter
{
public:
    void open_element(const string& name);
    void close_element();

    void begin_array(const string& name);
    void end_array();
    void begin_array_object();
    void end_array_object();
    void add_field(const string& name, const string& value);

    string c_str(int indent = 2) const;

private:
    Json                root;
    vector<Json*>  stack;  // path of containers currently open
    vector<string> name_stack; // for opened named scalar/object
};
void add_json_field(JsonWriter& writer,
                    const string& name,
                    const string& value);

void write_json(JsonWriter& writer,
                initializer_list<pair<const char*, string>> props);
float       read_json_type   (const Json* root, const string& field);
long        read_json_index  (const Json* root, const string& field);
bool        read_json_bool   (const Json* root, const string& field);
string read_json_string (const Json* root, const string& field);

string read_json_string_fallback(const Json* root,
                                      initializer_list<string> names);

const Json* require_json_field(const Json* root, const string& field);

template<typename Func>
void for_json_items(const Json* parent, const char* tag, long count, Func func)
{
    if (!parent || !parent->is_object())
        throw runtime_error(string("Missing JSON parent for: ") + tag);

    const Json* arr = parent->find(tag);
    if (!arr || !arr->is_array() || long(arr->array_value.size()) != count)
        throw runtime_error(string("Missing or wrong-size JSON array: ") + tag);

    for (long i = 0; i < count; i++)
        func(i, &arr->array_value[size_t(i)]);
}

JsonDocument load_json_file(const filesystem::path& file_name);
const Json*  get_json_root (const JsonDocument& document, const string& tag);

}
