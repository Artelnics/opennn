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

inline const string& scaler_method_to_string(ScalerMethod method)
{
    static const string none_str = "None";
    static const string minmax_str = "MinimumMaximum";
    static const string meanstd_str = "MeanStandardDeviation";
    static const string std_str = "StandardDeviation";
    static const string log_str = "Logarithm";
    static const string imgminmax_str = "ImageMinMax";

    switch(method)
    {
    case ScalerMethod::MinimumMaximum:       return minmax_str;
    case ScalerMethod::MeanStandardDeviation: return meanstd_str;
    case ScalerMethod::StandardDeviation:    return std_str;
    case ScalerMethod::Logarithm:            return log_str;
    case ScalerMethod::ImageMinMax:          return imgminmax_str;
    default:                                 return none_str;
    }
}

inline ScalerMethod string_to_scaler_method(const string& name)
{
    if(name == "MinimumMaximum")       return ScalerMethod::MinimumMaximum;
    if(name == "MeanStandardDeviation") return ScalerMethod::MeanStandardDeviation;
    if(name == "StandardDeviation")    return ScalerMethod::StandardDeviation;
    if(name == "Logarithm")            return ScalerMethod::Logarithm;
    if(name == "ImageMinMax")          return ScalerMethod::ImageMinMax;
    if(name == "None")                 return ScalerMethod::None;

    throw runtime_error("Unknown scaler method: " + name);
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

inline const string& variable_role_to_string(VariableRole role)
{
    static const string none_str = "None";
    static const string input_str = "Input";
    static const string target_str = "Target";
    static const string decoder_str = "Decoder";
    static const string input_target_str = "InputTarget";
    static const string time_str = "Time";

    switch(role)
    {
    case VariableRole::Input:       return input_str;
    case VariableRole::Target:      return target_str;
    case VariableRole::Decoder:     return decoder_str;
    case VariableRole::InputTarget: return input_target_str;
    case VariableRole::Time:        return time_str;
    default:                        return none_str;
    }
}

inline VariableRole string_to_variable_role(const string& name)
{
    if(name == "Input")       return VariableRole::Input;
    if(name == "Target")      return VariableRole::Target;
    if(name == "Decoder")     return VariableRole::Decoder;
    if(name == "InputTarget") return VariableRole::InputTarget;
    if(name == "Time")        return VariableRole::Time;
    if(name == "None" || name == "Id") return VariableRole::None;

    throw runtime_error("Unknown variable role: " + name);
}

inline bool role_matches(VariableRole actual, VariableRole query)
{
    if(actual == query) return true;
    if(actual == VariableRole::InputTarget && (query == VariableRole::Input || query == VariableRole::Target))
        return true;
    return false;
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
