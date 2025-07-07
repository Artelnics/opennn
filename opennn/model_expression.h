//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M O D E L   E X P R E S S I O N   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef MODELEXPRESSION_H
#define MODELEXPRESSION_H

#include "dataset.h"

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
    void auto_association_c() const;
    string get_expression_c() const;

    // python
    string write_header_python() const;
    string write_subheader_python() const;
    string get_expression_python() const;

    // php
    string write_header_api() const;
    string write_subheader_api() const;
    void autoassociation_api() const;
    string logistic_api() const;
    string relu_api() const;
    string exponential_linear_api() const;
    string scaled_exponential_linear_api() const;
    string get_expression_api() const;

    // javascript
    string autoassociaton_javascript() const;
    string logistic_javascript() const;
    string relu_javascript() const;
    string exponential_linear_javascript() const;
    string selu_javascript() const;
    string header_javascript() const;
    string subheader_javascript() const;
    string get_expression_javascript(const vector<Dataset::RawVariable>& ) const;

    // other functions
    string replace_reserved_keywords(string&) const;
    vector<string> fix_get_expression_outputs(const string& ,const vector<string>& ,const ProgrammingLanguage&) const;
    vector<string> fix_input_names(vector<string>&) const;
    vector<string> fix_output_names(vector<string>& ) const;

    void save_python(const filesystem::path&) const;
    void save_c(const filesystem::path&) const;
    void save_javascript(const filesystem::path&, const vector<Dataset::RawVariable>&) const;
    void save_api(const filesystem::path&) const;

protected:
    const NeuralNetwork* neural_network = nullptr;
};

}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
