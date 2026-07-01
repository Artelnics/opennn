//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   J S O N   M I N I M A L   S U P P O R T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once
#ifndef OPENNN_JSON_H_
#define OPENNN_JSON_H_

#include <filesystem>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <format>

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

    bool is_null()   const noexcept { return kind == Kind::Null; }
    bool is_bool()   const noexcept { return kind == Kind::Bool; }
    bool is_number() const noexcept { return kind == Kind::Number; }
    bool is_string() const noexcept { return kind == Kind::String; }
    bool is_array()  const noexcept { return kind == Kind::Array; }
    bool is_object() const noexcept { return kind == Kind::Object; }
    bool         has(const string&) const;
    const Json*  find(const string&) const;
    const Json&  at(const string&) const;
    Json&        operator[](const string&);
    Json& set(const string&, Json);
    void push_back(Json);
    string as_string() const;
    long long   as_long()   const;
    double      as_double() const;
    bool        as_bool()   const;
    static Json  parse(const string&);
    string  dump(int indent = 2) const;
};

class JsonDocument
{
public:
    Json root;

    void load(const filesystem::path&);
    void save(const filesystem::path&, int indent = 2) const;
    const Json* first_child(const string&) const;
    const Json* first_child() const noexcept { return &root; }
    static JsonDocument wrap(const string&, Json);
};

class JsonWriter
{
public:
    void open_element(const string&);
    void close_element();

    void begin_array(const string&);
    void end_array();
    void begin_array_object();
    void end_array_object();
    void add_field(const string&, const string&);

    string c_str(int indent = 2) const;

private:
    void pop_scope();

    Json                root;
    vector<Json*>  stack;
    vector<string> name_stack;
};
void add_json_field(JsonWriter&,
                    const string&,
                    const string&);

void save_json_file(const filesystem::path&, const JsonWriter&);

template <typename Serializable>
void save_json_file(const filesystem::path& file_name, const Serializable& serializable)
{
    JsonWriter writer;
    serializable.to_JSON(writer);
    save_json_file(file_name, writer);
}

void write_json(JsonWriter&,
                initializer_list<pair<const char*, string>>);
float       read_json_float   (const Json*, const string&);
long long   read_json_index  (const Json*, const string&);
bool        read_json_bool   (const Json*, const string&);
string read_json_string (const Json*, const string&);

string read_json_string_fallback(const Json*,
                                      initializer_list<string>);

const Json* require_json_field(const Json*, const string&);

template<typename Func>
void for_json_items(const Json* parent, const char* tag, long count, Func func)
{
    throw_if(!parent || !parent->is_object(),
             format("Missing JSON parent for: {}", tag));

    const Json* arr = parent->find(tag);
    throw_if(!arr || !arr->is_array() || long(arr->array_value.size()) != count,
             format("Missing or wrong-size JSON array: {}", tag));

    for (long i = 0; i < count; i++)
        func(i, &arr->array_value[size_t(i)]);
}

JsonDocument load_json_file(const filesystem::path&);
const Json*  get_json_root (const JsonDocument&, const string&);

}

#endif // OPENNN_JSON_H_
