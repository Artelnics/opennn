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
    string role = "None";
    VariableType type = VariableType::None;
    vector<string> categories;
    string scaler = "None";

    // Methods
    string get_role() const;
    string get_type_string() const;
    Index get_categories_number() const;

    void set_scaler(const string&);
    void set_role(const string&);
    void set_type(const string&);
    void set_categories(const vector<string>&);

    void from_XML(const XMLDocument&);
    void to_XML(XMLPrinter&) const;

    bool is_binary() const { return type == VariableType::Binary; }
    bool is_categorical() const { return type == VariableType::Categorical; }
    bool is_used() const { return !(role == "None" || role == "Time"); }

    void print() const;

    vector<string> get_names() const;

};

}
