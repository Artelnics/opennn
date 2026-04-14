//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   V A R I A B L E    S T R U C T U R E  H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pch.h"

#pragma once

namespace opennn
{

enum class VariableType { None, Numeric, Binary, Categorical, DateTime, Constant };

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
    if(name == "Id") return VariableRole::None;
    return variable_role_map().from_string(name);
}

inline bool role_matches(VariableRole actual, VariableRole query)
{
    return actual == query
        || (actual == VariableRole::InputTarget && (query == VariableRole::Input || query == VariableRole::Target));
}

struct Variable
{
    Variable(const string& = string(),
             const string& = "None",
             const VariableType& = VariableType::Numeric,
             const string& = "MeanStandardDeviation",
             const vector<string>& = vector<string>());

    void set(const string& = string(),
             const string& = "None",
             const VariableType& = VariableType::Numeric,
             const string& = "MeanStandardDeviation",
             const vector<string>& = vector<string>());

    string name;
    VariableRole role = VariableRole::None;
    VariableType type = VariableType::None;
    vector<string> categories;
    ScalerMethod scaler = ScalerMethod::None;

    // Methods
    const string& get_role() const { return variable_role_to_string(role); }
    VariableRole get_role_type() const { return role; }
    const string& get_scaler() const { return scaler_method_to_string(scaler); }
    ScalerMethod get_scaler_type() const { return scaler; }
    string get_type_string() const;
    Index get_categories_number() const;

    void set_scaler(const string& s) { scaler = string_to_scaler_method(s); }
    void set_scaler(ScalerMethod s) { scaler = s; }
    void set_role(const string& r) { role = string_to_variable_role(r); }
    void set_role(VariableRole r) { role = r; }
    void set_type(const string&);
    void set_categories(const vector<string>& c) { categories = c; }

    void from_XML(const XmlDocument&);
    void to_XML(XmlPrinter&) const;

    bool is_binary() const { return type == VariableType::Binary; }
    bool is_categorical() const { return type == VariableType::Categorical; }
    bool is_used() const { return role != VariableRole::None && role != VariableRole::Time; }

    vector<string> get_names() const;

};

}
