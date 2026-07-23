//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   V A R I A B L E    S T R U C T U R E  H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"
#include "enum_map.h"

namespace opennn
{

enum class VariableType { None, Numeric, Binary, Integer, Categorical, DateTime, Constant };

inline const EnumMap<VariableType>& variable_type_map()
{
    static const vector<pair<VariableType, string>> entries = {
        {VariableType::None,        "None"},
        {VariableType::Numeric,     "Numeric"},
        {VariableType::Binary,      "Binary"},
        {VariableType::Integer,     "Integer"},
        {VariableType::Categorical, "Categorical"},
        {VariableType::DateTime,    "DateTime"},
        {VariableType::Constant,    "Constant"}
    };
    static const EnumMap<VariableType> map{entries};
    return map;
}

inline const string& variable_type_to_string(VariableType type)
{
    return variable_type_map().to_string(type);
}

inline VariableType string_to_variable_type(const string& name)
{
    return variable_type_map().from_string(name);
}

enum class ScalerMethod
{
    None,
    MinimumMaximum,
    MeanStandardDeviation,
    StandardDeviation,
    Logarithm,
    ImageMinMax
};

inline const EnumMap<ScalerMethod>& scaler_method_map()
{
    static const vector<pair<ScalerMethod, string>> entries = {
        {ScalerMethod::None,                 "None"},
        {ScalerMethod::MinimumMaximum,       "MinimumMaximum"},
        {ScalerMethod::MeanStandardDeviation, "MeanStandardDeviation"},
        {ScalerMethod::StandardDeviation,    "StandardDeviation"},
        {ScalerMethod::Logarithm,            "Logarithm"},
        {ScalerMethod::ImageMinMax,          "ImageMinMax"}
    };
    static const EnumMap<ScalerMethod> map{entries};
    return map;
}

inline const string& scaler_method_to_string(ScalerMethod method)
{
    return scaler_method_map().to_string(method);
}

inline ScalerMethod string_to_scaler_method(const string& name)
{
    return scaler_method_map().from_string(name);
}

enum class VariableRole
{
    None,
    Input,
    Target,
    Decoder,
    InputTarget,
    Time
};

inline const EnumMap<VariableRole>& variable_role_map()
{
    static const vector<pair<VariableRole, string>> entries = {
        {VariableRole::None,        "None"},
        {VariableRole::Input,       "Input"},
        {VariableRole::Target,      "Target"},
        {VariableRole::Decoder,     "Decoder"},
        {VariableRole::InputTarget, "InputTarget"},
        {VariableRole::Time,        "Time"}
    };
    static const EnumMap<VariableRole> map{entries};
    return map;
}

inline const string& variable_role_to_string(VariableRole role)
{
    return variable_role_map().to_string(role);
}

inline VariableRole string_to_variable_role(const string& name)
{
    if (name == "Id") return VariableRole::None;
    return variable_role_map().from_string(name);
}

inline bool role_applies_to(VariableRole actual, VariableRole query)
{
    return actual == query
        || (actual == VariableRole::InputTarget && (query == VariableRole::Input || query == VariableRole::Target));
}

struct Variable
{
    Variable(const string& = {},
             const string& = "None",
             const VariableType& = VariableType::Numeric,
             const string& = "MeanStandardDeviation",
             const vector<string>& = {});

    void set(const string& = {},
             const string& = "None",
             const VariableType& = VariableType::Numeric,
             const string& = "MeanStandardDeviation",
             const vector<string>& = {});

    string name;
    VariableRole role = VariableRole::None;
    VariableType type = VariableType::None;
    vector<string> categories;
    ScalerMethod scaler = ScalerMethod::None;

    // Number of identical features this variable spans (e.g. the pixels of an
    // image block or the tokens of a sequence). Categorical variables span one
    // feature per category instead.
    Index features = 1;
    const string& get_role() const { return variable_role_to_string(role); }
    VariableRole get_role_type() const noexcept { return role; }
    const string& get_scaler() const { return scaler_method_to_string(scaler); }
    ScalerMethod get_scaler_type() const noexcept { return scaler; }
    const string& get_type_string() const;
    Index get_categories_number() const;

    void set_scaler(const string& new_scaler) { scaler = string_to_scaler_method(new_scaler); }
    void set_scaler(ScalerMethod new_scaler) { scaler = new_scaler; }
    void set_role(const string& new_role) { role = string_to_variable_role(new_role); }
    void set_role(VariableRole new_role) { role = new_role; }
    void set_type(const string&);
    void set_categories(const vector<string>& new_categories) { categories = new_categories; }

    void to_JSON(JsonWriter&) const;

    bool is_binary() const noexcept { return type == VariableType::Binary; }
    bool is_integer() const noexcept { return type == VariableType::Integer; }
    bool is_categorical() const noexcept { return type == VariableType::Categorical; }
    bool is_used() const noexcept { return role != VariableRole::None && role != VariableRole::Time; }

    Index get_feature_count() const { return is_categorical() ? get_categories_number() : features; }

    vector<string> get_names() const;

};

vector<string> get_variable_feature_names(const vector<Variable>&);

// Per-variable feature counts and their sum, for a list of variables.
inline vector<Index> get_feature_dimensions(const vector<Variable>& variables)
{
    vector<Index> dimensions;
    dimensions.reserve(variables.size());
    for (const Variable& variable : variables)
        dimensions.push_back(variable.get_feature_count());
    return dimensions;
}

inline Index get_features_number(const vector<Variable>& variables)
{
    Index count = 0;
    for (const Variable& variable : variables)
        count += variable.get_feature_count();
    return count;
}

}
