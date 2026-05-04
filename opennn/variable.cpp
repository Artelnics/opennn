//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   V A R I A B L E    S T R U C T U R E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "variable.h"
#include "string_utilities.h"
#include "tensor_utilities.h"

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

void Variable::set_type(const string& new_variable_type)
{
    if (new_variable_type == "Numeric")
        type = VariableType::Numeric;
    else if (new_variable_type == "Binary")
        type = VariableType::Binary;
    else if (new_variable_type == "Categorical")
        type = VariableType::Categorical;
    else if (new_variable_type == "DateTime")
        type = VariableType::DateTime;
    else if (new_variable_type == "Constant")
        type = VariableType::Constant;
    else if (new_variable_type == "None")
        type = VariableType::None;
    else
        throw runtime_error("Variable type is not valid (" + new_variable_type + ").\n");
}

string Variable::get_type_string() const
{
    switch (type)
    {
    case VariableType::None:
        return "None";
    case VariableType::Numeric:
        return "Numeric";
    case VariableType::Constant:
        return "Constant";
    case VariableType::Binary:
        return "Binary";
    case VariableType::Categorical:
        return "Categorical";
    case VariableType::DateTime:
        return "DateTime";
    default:
        return "Unknown";
    }
}

Index Variable::get_categories_number() const
{
    return ssize(categories);
}

void Variable::from_JSON(const JsonDocument& document)
{
    name = read_json_string(document.first_child(), "Name");
    set_scaler(read_json_string(document.first_child(), "Scaler"));
    set_role(read_json_string(document.first_child(), "Role"));
    set_type(read_json_string(document.first_child(), "Type"));

    if (type == VariableType::Categorical)
    {
        const string categories_text = read_json_string(document.first_child(), "Categories");
        categories = get_tokens(categories_text, ";");
    }
}

void Variable::to_JSON(JsonWriter& printer) const
{
    write_json(printer, {
        {"Name", name},
        {"Scaler", get_scaler()},
        {"Role", get_role()},
        {"Type", get_type_string()}
    });

    if (type == VariableType::Categorical || type == VariableType::Binary)
        add_json_field(printer, "Categories", vector_to_string(categories, ";"));
}

vector<string> Variable::get_names() const
{
    return is_categorical()
       ? categories
       : vector<string>{name};
}

}
