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

#include <cstddef>
#include <filesystem>
#include <format>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace opennn
{

class Json
{
public:
    enum class Kind { Null, Bool, Number, String, Array, Object };

    Kind                                    kind = Kind::Null;
    bool                                    bool_value = false;
    double                                  number_value = 0.0;
    std::string                              string_value;
    std::vector<Json>                        array_value;
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
    Json(const std::string& s)      : kind(Kind::String), string_value(s) {}
    Json(std::string_view s)        : kind(Kind::String), string_value(s) {}

    static Json make_object();
    static Json make_array();

    bool is_null()   const noexcept { return kind == Kind::Null; }
    bool is_bool()   const noexcept { return kind == Kind::Bool; }
    bool is_number() const noexcept { return kind == Kind::Number; }
    bool is_string() const noexcept { return kind == Kind::String; }
    bool is_array()  const noexcept { return kind == Kind::Array; }
    bool is_object() const noexcept { return kind == Kind::Object; }
    bool         has(std::string_view) const;
    const Json*  find(std::string_view) const;
    const Json&  at(std::string_view) const;
    Json&        operator[](std::string_view);
    Json& set(std::string_view, Json);
    void push_back(Json);
    std::string as_string() const;
    long long   as_long()   const;
    double      as_double() const;
    bool        as_bool()   const;
    static Json parse(std::string_view);
    std::string dump(int indent = 2) const;
};

class JsonDocument
{
public:
    Json root;

    void load(const std::filesystem::path&);
    void save(const std::filesystem::path&, int indent = 2) const;
    const Json* first_child(std::string_view) const;
    const Json* first_child() const noexcept { return &root; }
    static JsonDocument wrap(std::string_view, Json);
};

class JsonWriter
{
public:
    void open_element(std::string_view);
    void close_element();

    void begin_array(std::string_view);
    void end_array();
    void begin_array_object();
    void end_array_object();
    void add_field(std::string_view, Json);

    template <typename Value>
    void add_field(std::string_view name, Value&& value)
    {
        add_field(name, Json(std::forward<Value>(value)));
    }

    std::string c_str(int indent = 2) const;

private:
    void pop_scope();

    Json               root;
    std::vector<Json*> stack;
};
void add_json_field(JsonWriter&,
                    std::string_view,
                    Json);

template <typename Value>
void add_json_field(JsonWriter& writer, std::string_view name, Value&& value)
{
    writer.add_field(name, std::forward<Value>(value));
}

void save_json_file(const std::filesystem::path&, const JsonWriter&);

template <typename Serializable>
void save_json_file(const std::filesystem::path& file_name, const Serializable& serializable)
{
    JsonWriter writer;
    serializable.to_JSON(writer);
    save_json_file(file_name, writer);
}

void write_json(JsonWriter&,
                std::initializer_list<std::pair<const char*, Json>>);
float                    read_json_float  (const Json*, std::string_view);
long long                read_json_index  (const Json*, std::string_view);
bool                     read_json_bool   (const Json*, std::string_view);
std::string              read_json_string (const Json*, std::string_view);
std::vector<std::string> read_json_strings(const Json*, std::string_view);

std::string read_json_string_fallback(const Json*,
                                      std::initializer_list<std::string_view>);

const Json* require_json_field(const Json*, std::string_view);

template <typename Range>
Json json_array(const Range& values)
{
    Json array = Json::make_array();
    if constexpr (requires { std::size(values); })
        array.array_value.reserve(std::size(values));
    for (const auto& value : values)
        array.push_back(Json(value));
    return array;
}

template<typename Func>
void for_json_items(const Json* parent, const char* tag, std::size_t count, Func func)
{
    if (!parent || !parent->is_object())
        throw std::runtime_error(std::format("Missing JSON parent for: {}", tag));

    const Json* const arr = parent->find(tag);
    if (!arr || !arr->is_array() || arr->array_value.size() != count)
        throw std::runtime_error(std::format("Missing or wrong-size JSON array: {}", tag));

    for (std::size_t i = 0; i < count; ++i)
        func(i, &arr->array_value[i]);
}

JsonDocument load_json_file(const std::filesystem::path&);
const Json*  get_json_root (const JsonDocument&, std::string_view);

}

#endif
