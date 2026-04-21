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
class Layer;
class Bounding;
class Unscaling;
class Recurrent;
template<int Rank> class Scaling;
template<int Rank> class Dense;

class ModelExpression
{
public:

    enum class ProgrammingLanguage{C, Python, JavaScript, PHP};

    ModelExpression(const NeuralNetwork*);

    string build_expression() const;

    void save(const filesystem::path&, ProgrammingLanguage) const;

private:

    const NeuralNetwork* neural_network = nullptr;

    string get_expression_c() const;
    string get_expression_python() const;
    string get_expression_php() const;
    string get_expression_javascript() const;

    static string get_layer_expression(const Layer&, const vector<string>&, const vector<string>&);

    void emit_c_prelude(ostringstream&) const;
    void emit_c_activations(ostringstream&, const string& expression) const;
    void emit_c_calculate_outputs(ostringstream&, const string& expression, const vector<string>& lines, bool has_softmax) const;
    void emit_c_main(ostringstream&) const;

    void emit_php_prelude(ostringstream&) const;
    void emit_php_activations(ostringstream&, const string& expression) const;
    void emit_php_inputs_setup(ostringstream&) const;
    void emit_php_body(ostringstream&, const vector<string>& lines, bool has_softmax) const;
    void emit_php_response(ostringstream&) const;

    void emit_python_prelude(ostringstream&) const;
    void emit_python_class_header(ostringstream&) const;
    void emit_python_activations(ostringstream&, const string& expression) const;
    void emit_python_calculate_outputs(ostringstream&, const vector<string>& lines, bool has_softmax) const;
    void emit_python_batch_and_main(ostringstream&) const;

    void emit_js_prelude(ostringstream&) const;
    void emit_js_inputs_html(ostringstream&) const;
    void emit_js_outputs_html(ostringstream&, bool use_category_select) const;
    void emit_js_runtime(ostringstream&, const string& expression, const vector<string>& lines, bool has_softmax, bool use_category_select) const;

    static vector<string> split_expression_lines(const string&);
    static void rename_spaced_var_definitions(vector<string>&);
    static vector<string> prepare_body_lines(const string& expression);
    static vector<string> fix_names(const vector<string>&, const string& default_prefix);
    static vector<string> fix_get_expression_outputs(const string&, const vector<string>&, const ProgrammingLanguage&);
    static void apply_name_mapping(string&, const vector<string>& original, const vector<string>& mapped);
    static string process_body_line(const string&, const vector<string>& input_names, const vector<string>& fixed_input_names);
    static string replace_reserved_keywords(const string&);

    struct ActivationBodies
    {
        const char* c;
        const char* javascript;
        const char* python;
        const char* php;
    };

    static const vector<pair<string, ActivationBodies>>& activation_table();

    static string write_bounding_expression(const Bounding&, const vector<string>& , const vector<string>&);
    static string write_scaling_expression(const Scaling<2>&, const vector<string>&, const vector<string>&);
    static string write_unscaling_expression(const Unscaling&, const vector<string>&, const vector<string>&);
    static string write_recurrent_expression(const Recurrent&, const vector<string>&, const vector<string>&);
    static string write_dense_expression(const Dense<2>&, const vector<string>&, const vector<string>&);
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
