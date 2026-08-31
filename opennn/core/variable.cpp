//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   V A R I A B L E    S T R U C T U R E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/core/variable.h"
#include "opennn/core/string_utilities.h"
#include "opennn/core/tensor_types.h"

namespace opennn
{

Variable::Variable(const string& new_name, const string& new_variable_role, const VariableType& new_type, const string& new_scaler, const vector<string>& new_categories)
{
    set(new_name, new_variable_role, new_type, new_scaler, new_categories);
}

void Variable::set(const string& new_name, const string& new_variable_role, const VariableType& new_type, const string& new_scaler, const vector<string>& new_categories)
{
    name = new_name;
    role = string_to_variable_role(new_variable_role);
    type = new_type;
    scaler = string_to_scaler_method(new_scaler);
    categories = new_categories;
}

void Variable::to_JSON(JsonWriter& printer) const
{
    write_json(printer, {
        {"Name", name},
        {"Scaler", get_scaler()},
        {"Role", get_role()},
        {"Type", get_type_string()}
    });

    if (features > 1)
        add_json_field(printer, "Features", features);

    if (is_one_of(type, VariableType::Categorical, VariableType::Binary))
        add_json_field(printer, "Categories", vector_to_string(categories, ";"));
}

vector<string> Variable::get_names() const
{
    if (is_categorical())
        return categories;

    if (features == 1)
        return {name};

    vector<string> names;
    names.reserve(size_t(features));

    for (Index i = 0; i < features; ++i)
        names.push_back(format("{}_{}", name, i + 1));

    return names;
}

vector<string> get_variable_feature_names(const vector<Variable>& variables)
{
    vector<string> feature_names;
    feature_names.reserve(size_t(get_features_number(variables)));

    for (const Variable& variable : variables)
    {
        vector<string> names = variable.get_names();
        feature_names.insert(feature_names.end(),
                             make_move_iterator(names.begin()),
                             make_move_iterator(names.end()));
    }

    return feature_names;
}

vector<pair<string, Index>> get_variable_columns(const vector<Variable>& variables)
{
    vector<pair<string, Index>> columns;

    Index feature_index = 0;

    for (const Variable& variable : variables)
    {
        const Index span = variable.get_feature_count();

        if (span == 1)
            columns.emplace_back(variable.name, feature_index);

        feature_index += span;
    }

    return columns;
}

vector<pair<Index, Index>> get_categorical_blocks(const vector<Variable>& variables)
{
    vector<pair<Index, Index>> blocks;

    Index feature_index = 0;

    for (const Variable& variable : variables)
    {
        const Index span = variable.get_feature_count();

        if (variable.is_categorical() && span > 1)
            blocks.emplace_back(feature_index, span);

        feature_index += span;
    }

    return blocks;
}
}
