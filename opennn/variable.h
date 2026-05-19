//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   V A R I A B L E    S T R U C T U R E  H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pch.h"
#include "tensor_utilities.h"
#include "enum_map.h"

#pragma once

namespace opennn
{

/// @brief Data type of a dataset Variable.
enum class VariableType { None, Numeric, Binary, Categorical, DateTime, Constant };

/// @brief Returns the bidirectional string/enum map for VariableType.
inline const EnumMap<VariableType>& variable_type_map()
{
    static const vector<pair<VariableType, string>> entries = {
        {VariableType::None,        "None"},
        {VariableType::Numeric,     "Numeric"},
        {VariableType::Binary,      "Binary"},
        {VariableType::Categorical, "Categorical"},
        {VariableType::DateTime,    "DateTime"},
        {VariableType::Constant,    "Constant"}
    };
    static const EnumMap<VariableType> map{entries};
    return map;
}

/// @brief Returns the canonical string name for a VariableType.
inline const string& variable_type_to_string(VariableType type)
{
    return variable_type_map().to_string(type);
}

/// @brief Parses a string into the matching VariableType enumerator.
inline VariableType string_to_variable_type(const string& name)
{
    return variable_type_map().from_string(name);
}

/// @brief Feature scaling strategy applied to a Variable before training.
enum class ScalerMethod
{
    None,
    MinimumMaximum,
    MeanStandardDeviation,
    StandardDeviation,
    Logarithm,
    ImageMinMax
};

/// @brief Returns the bidirectional string/enum map for ScalerMethod.
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

/// @brief Returns the canonical string name for a ScalerMethod.
inline const string& scaler_method_to_string(ScalerMethod method)
{
    return scaler_method_map().to_string(method);
}

/// @brief Parses a string into the matching ScalerMethod enumerator.
inline ScalerMethod string_to_scaler_method(const string& name)
{
    return scaler_method_map().from_string(name);
}

/// @brief Role a Variable plays in a dataset (input feature, target, decoder, time axis, etc.).
enum class VariableRole
{
    None,
    Input,
    Target,
    Decoder,
    InputTarget,
    Time
};

/// @brief Returns the bidirectional string/enum map for VariableRole.
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

/// @brief Returns the canonical string name for a VariableRole.
inline const string& variable_role_to_string(VariableRole role)
{
    return variable_role_map().to_string(role);
}

/// @brief Parses a string into a VariableRole; "Id" is mapped to VariableRole::None.
inline VariableRole string_to_variable_role(const string& name)
{
    if (name == "Id") return VariableRole::None;
    return variable_role_map().from_string(name);
}

/// @brief Returns true if @p actual satisfies @p query, treating InputTarget as Input or Target.
inline bool role_matches(VariableRole actual, VariableRole query)
{
    return actual == query
        || (actual == VariableRole::InputTarget && (query == VariableRole::Input || query == VariableRole::Target));
}

/// @brief Single dataset column descriptor: name, role, type, scaler, and optional categories.
struct Variable
{
    /// @brief Constructs a Variable with optional name, role, type, scaler and category list.
    /// @param name Variable name.
    /// @param role Role string (parsed via string_to_variable_role).
    /// @param type Variable data type.
    /// @param scaler Scaler method name (parsed via string_to_scaler_method).
    /// @param categories Category labels for categorical variables.
    Variable(const string& = {},
             const string& = "None",
             const VariableType& = VariableType::Numeric,
             const string& = "MeanStandardDeviation",
             const vector<string>& = {});

    /// @brief Resets the Variable fields to the supplied values.
    /// @param name Variable name.
    /// @param role Role string (parsed via string_to_variable_role).
    /// @param type Variable data type.
    /// @param scaler Scaler method name (parsed via string_to_scaler_method).
    /// @param categories Category labels for categorical variables.
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
    const string& get_role() const { return variable_role_to_string(role); }
    VariableRole get_role_type() const { return role; }
    const string& get_scaler() const { return scaler_method_to_string(scaler); }
    ScalerMethod get_scaler_type() const { return scaler; }
    /// @brief Returns the canonical string name of the Variable type.
    const string& get_type_string() const;
    /// @brief Returns the number of categories for categorical variables (zero otherwise).
    Index get_categories_number() const;

    void set_scaler(const string& new_scaler) { scaler = string_to_scaler_method(new_scaler); }
    void set_scaler(ScalerMethod new_scaler) { scaler = new_scaler; }
    void set_role(const string& new_role) { role = string_to_variable_role(new_role); }
    void set_role(VariableRole new_role) { role = new_role; }
    /// @brief Sets the Variable type from its canonical string name.
    void set_type(const string&);
    void set_categories(const vector<string>& new_categories) { categories = new_categories; }

    /// @brief Loads Variable fields from a JSON document.
    void from_JSON(const JsonDocument&);
    /// @brief Writes Variable fields to a JSON writer.
    void to_JSON(JsonWriter&) const;

    /// @brief Returns true if the Variable type is Binary.
    bool is_binary() const { return type == VariableType::Binary; }
    /// @brief Returns true if the Variable type is Categorical.
    bool is_categorical() const { return type == VariableType::Categorical; }
    /// @brief Returns true if the Variable has an active role other than Time.
    bool is_used() const { return role != VariableRole::None && role != VariableRole::Time; }

    /// @brief Returns the number of features generated by this Variable (categories or one).
    Index feature_count() const { return is_categorical() ? get_categories_number() : 1; }

    /// @brief Returns the expanded feature names (one per category for categorical variables).
    vector<string> get_names() const;

};

}
