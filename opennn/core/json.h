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
#include <variant>
#include <vector>

namespace opennn
{

class Json
{
public:
    using Array = std::vector<Json>;
    using Object = std::vector<std::pair<std::string, Json>>;

    enum class Kind { Null, Bool, Number, String, Array, Object };

    static Json make_object();
    static Json make_array();

    Json() = default;
    Json(bool new_value) : value(new_value) {}
    template <typename Int, typename = std::enable_if_t<std::is_integral_v<Int> && !std::is_same_v<Int, bool>>>
    Json(Int new_value) : value(double(new_value)) {}
    Json(double new_value) : value(new_value) {}
    Json(float new_value) : value(double(new_value)) {}
    Json(const char* new_value) : value(std::string(new_value)) {}
    Json(const std::string& new_value) : value(new_value) {}
    Json(std::string_view new_value) : value(std::string(new_value)) {}

    Kind get_kind() const noexcept { return Kind(value.index()); }
    bool is_null() const noexcept { return std::holds_alternative<std::monostate>(value); }
    bool is_bool() const noexcept { return std::holds_alternative<bool>(value); }
    bool is_number() const noexcept { return std::holds_alternative<double>(value); }
    bool is_string() const noexcept { return std::holds_alternative<std::string>(value); }
    bool is_array() const noexcept { return std::holds_alternative<Array>(value); }
    bool is_object() const noexcept { return std::holds_alternative<Object>(value); }
    Array& as_array();
    const Array& as_array() const;
    Object& as_object();
    const Object& as_object() const;
    bool         has(std::string_view key) const { return find(key) != nullptr; }
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

private:
    using Value = std::variant<std::monostate, bool, double, std::string, Array, Object>;

    Value value;
};

class JsonDocument
{
public:
    static JsonDocument wrap(std::string_view, Json);

    void load(const std::filesystem::path&);
    void save(const std::filesystem::path&, int indent = 2) const;
    void set_root(Json new_root) { root = std::move(new_root); }
    Json& get_root() noexcept { return root; }
    const Json& get_root() const noexcept { return root; }
    const Json* first_child(std::string_view name) const { return root.find(name); }
    const Json* first_child() const noexcept { return &root; }

private:
    Json root;
};

class JsonWriter
{
public:
    void open_element(std::string_view);
    void close_element() { pop_scope(); }

    void begin_array(std::string_view);
    void end_array() { pop_scope(); }
    void begin_array_object();
    void end_array_object() { pop_scope(); }
    void add_field(std::string_view, Json);

    template <typename Value>
    void add_field(std::string_view name, Value&& value)
    {
        add_field(name, Json(std::forward<Value>(value)));
    }

    std::string c_str(int indent = 2) const { return root.dump(indent); }

private:
    void pop_scope();

    Json               root;
    std::vector<Json*> stack;
};
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

// Overloads that keep the caller's value when the field is absent. Without
// them every optional field costs a `if (el->has("X")) set_x(read_json_x(...))`
// pair, because the no-fallback readers return 0/false/"" for a missing field
// and would otherwise overwrite a perfectly good default.
float       read_json_float (const Json*, std::string_view, float fallback);
long long   read_json_index (const Json*, std::string_view, long long fallback);
bool        read_json_bool  (const Json*, std::string_view, bool fallback);
std::string read_json_string(const Json*, std::string_view, std::string_view fallback);

std::string read_json_string_fallback(const Json*,
                                      std::initializer_list<std::string_view>);

const Json* require_json_field(const Json*, std::string_view);

template <typename Range>
Json json_array(const Range& values)
{
    Json array = Json::make_array();
    if constexpr (requires { std::size(values); })
        array.as_array().reserve(std::size(values));
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
    if (!arr || !arr->is_array() || arr->as_array().size() != count)
        throw std::runtime_error(std::format("Missing or wrong-size JSON array: {}", tag));

    for (std::size_t i = 0; i < count; ++i)
        func(i, &arr->as_array()[i]);
}

JsonDocument load_json_file(const std::filesystem::path&);
const Json*  get_json_root (const JsonDocument&, std::string_view);

}

#endif
