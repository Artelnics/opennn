//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M O D E L   E X P R E S S I O N   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "variable.h"

namespace opennn
{

class NeuralNetwork;

class ModelExpression
{
public:

    enum class ProgrammingLanguage{C, Python, JavaScript, PHP};

    ModelExpression(const NeuralNetwork*);

    // c
    string write_comments_c() const;
    string write_logistic_c() const;
    string write_relu_c() const;
    string write_exponential_linear_c() const;
    string write_selu_c() const;
    string write_softmax_c() const;
    //void auto_association_c() const;
    string get_expression_c(const vector<Variable>&) const;

    // python
    string write_header_python() const;
    string write_subheader_python() const;
    string get_expression_python(const vector<Variable>&) const;

    // php
    string write_header_api() const;
    string write_subheader_api() const;
    string get_expression_api(const vector<Variable>&) const;

    // javascript
    //string autoassociaton_javascript() const;
    string logistic_javascript() const;
    string relu_javascript() const;
    string exponential_linear_javascript() const;
    string selu_javascript() const;
    string hyperbolic_tangent_javascript() const;
    string softmax_javascript() const;
    string header_javascript() const;
    string subheader_javascript() const;
    string get_expression_javascript(const vector<Variable>&) const;

    // other functions
    string replace_reserved_keywords(const string&) const;
    vector<string> fix_get_expression_outputs(const string& ,const vector<string>& ,const ProgrammingLanguage&) const;
    vector<string> fix_feature_names(const vector<string>&) const;
    vector<string> fix_output_names(const vector<string>& ) const;

    void save_python(const filesystem::path&, const vector<Variable>&) const;
    void save_c(const filesystem::path&, const vector<Variable>&) const;
    void save_javascript(const filesystem::path&, const vector<Variable>&) const;
    void save_api(const filesystem::path&, const vector<Variable>&) const;

protected:
    const NeuralNetwork* neural_network = nullptr;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
