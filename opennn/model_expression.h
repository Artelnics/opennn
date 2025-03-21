//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M O D E L   E X P R E S S I O N   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef MODELEXPRESSION_H
#define MODELEXPRESSION_H

#include "strings_utilities.h"
#include "neural_network.h"

//#include "tensors.h"
//#include <unordered_map>

namespace opennn
{

class ModelExpression
{
public:

    enum class ProgrammingLanguage{C, Python, JavaScript, PHP};

    ModelExpression();

    // c
    string write_comments_c();
    string write_logistic_c();
    string write_relu_c();
    string write_exponential_linear_c();
    string write_selu_c();
    string write_hard_sigmoid_c();
    string write_soft_plus_c();
    string write_soft_sign_c();
    void lstm_c();
    void auto_association_c(const NeuralNetwork& );
    string get_expression_c(const NeuralNetwork& );

    // python
    string write_header_python();
    string write_subheader_python();
    string get_expression_python(const NeuralNetwork& );

    // php
    string write_header_api();
    void lstm_api();
    void autoassociation_api(const NeuralNetwork& );
    string logistic_api();
    string relu_api();
    string exponential_linear_api();
    string scaled_exponential_linear_api();
    string hard_sigmoid();
    string soft_plus();
    string soft_sign();
    string get_expression_api(const NeuralNetwork& );

    // javascript
    string autoassociaton_javascript(const NeuralNetwork& );
    string logistic_javascript();
    string relu_javascript();
    string exponential_linear_javascript();
    string selu_javascript();
    string hard_sigmoid_javascript();
    string soft_plus_javascript();
    string softsign_javascript();
    string header_javascript();
    string get_expression_javascript(const NeuralNetwork& );

    // other functions
    string replace_reserved_keywords(const string& );
    vector<string> fix_get_expression_outputs(const string& ,const vector<string>& ,const ProgrammingLanguage& );
    void fix_input_names(vector<string>& );
    void fix_output_names(vector<string>& );

    void save_expression(const string&, const ProgrammingLanguage&, const NeuralNetwork* );
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
