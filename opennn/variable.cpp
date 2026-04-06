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
#include "tinyxml2.h"
#include "pch.h"

namespace opennn
{

// Constructor
Variable::Variable(const string& new_name, const string& new_variable_role, const VariableType& new_type, const string& new_scaler, const vector<string>& new_categories)
{
    set(new_name, new_variable_role, new_type, new_scaler, new_categories);
}

// Set all members
void Variable::set(const string& new_name, const string& new_variable_role, const VariableType& new_type, const string& new_scaler, const vector<string>& new_categories)
{
    name = new_name;
    role = new_variable_role;
    type = new_type;
    scaler = new_scaler;
    categories = new_categories;
}

void Variable::set_scaler(const string& new_scaler)
{
    scaler = new_scaler;
}

void Variable::set_role(const string& new_variable_role)
{
    role = new_variable_role;
}

// Convert string to VariableType enum
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

void Variable::set_categories(const vector<string>& new_categories)
{
    categories = new_categories;
}

// Convert VariableType enum to string
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
        throw runtime_error("Unknown variable type");
    }
}

string Variable::get_role() const
{
    return role;
}

Index Variable::get_categories_number() const
{
    return static_cast<Index>(categories.size());
}

// XML Serialization
void Variable::from_XML(const XMLDocument& document)
{
    cout<<"FROM_XML CML SERIALIZATION)"<<endl;
    name = read_xml_string(document.FirstChildElement(), "Name");
    set_scaler(read_xml_string(document.FirstChildElement(), "Scaler"));
    set_role(read_xml_string(document.FirstChildElement(), "Role"));
    set_type(read_xml_string(document.FirstChildElement(), "Type"));

    if (type == VariableType::Categorical)
    {
        cout<<"ENTRA en CATEGORICAL"<<endl;

        const string categories_text = read_xml_string(document.FirstChildElement(), "Categories");
        categories = get_tokens(categories_text, ";");
    }
}

void Variable::to_XML(XMLPrinter& printer) const
{
    add_xml_element(printer, "Name", name);
    add_xml_element(printer, "Scaler", scaler);
    add_xml_element(printer, "Role", get_role());
    add_xml_element(printer, "Type", get_type_string());

    if (type == VariableType::Categorical || type == VariableType::Binary)
        add_xml_element(printer, "Categories", vector_to_string(categories, ";"));
}

// Console output
void Variable::print() const
{
    cout << "Variable" << endl
         << "Name: " << name << endl
         << "Role: " << get_role() << endl
         << "Type: " << get_type_string() << endl
         << "Scaler: " << scaler << endl;

    if (!categories.empty())
    {
        cout << "Categories: " << endl
             << categories;
    }

    cout << endl;
}

vector<string> Variable::get_names() const
{
    return is_categorical()
       ? categories
       : vector<string>{name};
}

}
