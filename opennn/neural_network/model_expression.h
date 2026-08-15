//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M O D E L   E X P R E S S I O N   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/core/configuration.h"
#include "opennn/core/opennn_types.h"

namespace opennn
{

class NeuralNetwork;
class Layer;

class ModelExpression
{
public:

    enum class ProgrammingLanguage{C, CEmbedded, Python, JavaScript, PHP};

    ModelExpression(const NeuralNetwork*);

    string build_expression() const;

    void save(const filesystem::path&, ProgrammingLanguage) const;

private:

    const NeuralNetwork* neural_network = nullptr;

    string get_expression_c() const;
    string get_expression_c_embedded() const;
    string get_expression_python() const;
    string get_expression_php() const;
    string get_expression_javascript() const;

    static string c_float_literal(float);

    vector<string> get_flat_input_names() const;

    void check_parameters_are_finite() const;

    void emit_c_prelude(ostringstream&) const;
    void emit_c_activations(ostringstream&, const string&) const;
    void emit_c_calculate_outputs(ostringstream&, const string&, const vector<string>&, bool) const;
    void emit_c_main(ostringstream&) const;

    void emit_php_prelude(ostringstream&) const;
    void emit_php_activations(ostringstream&, const string&) const;
    void emit_php_inputs_setup(ostringstream&) const;
    void emit_php_body(ostringstream&, const vector<string>&, bool) const;
    void emit_php_response(ostringstream&) const;

    void emit_python_prelude(ostringstream&) const;
    void emit_python_class_header(ostringstream&) const;
    void emit_python_activations(ostringstream&, const string&) const;
    void emit_python_calculate_outputs(ostringstream&, const string&, const vector<string>&, bool) const;
    void emit_python_batch_and_main(ostringstream&) const;

    void emit_js_prelude(ostringstream&) const;
    void emit_js_inputs_html(ostringstream&) const;
    void emit_js_outputs_html(ostringstream&, bool) const;
    void emit_js_runtime(ostringstream&, const string&, const vector<string>&, bool, bool) const;

    static vector<string> split_expression_lines(const string&);
    static void rename_spaced_var_definitions(vector<string>&);
    static vector<string> prepare_body_lines(const string&);
    static vector<string> fix_names(const vector<string>&, const string&);
    static vector<string> fix_output_names(const string&, const vector<string>&, const ProgrammingLanguage&);
    static void apply_name_mapping(string&, const vector<string>&, const vector<string>&);
    static string process_body_line(const string&, const vector<string>&, const vector<string>&);
    static string replace_reserved_keywords(const string&);

    struct LanguageSyntax
    {
        const char* body_indent = "";
        const char* body_declaration = "";
        bool declare_first_assignment_only = false;
        bool blank_short_lines = false;
        const char* softmax_declaration = "";
        const char* softmax_counter = "";
        const char* softmax_zero = "";
        const char* softmax_exp = "";
        string softmax_limit;
    };

    static LanguageSyntax language_syntax(ProgrammingLanguage, Index outputs_number = 0);
    static void emit_body_lines(ostringstream&, const vector<string>&, const LanguageSyntax&, const function<string(const string&)>&);
    static void emit_softmax_block(ostringstream&, const LanguageSyntax&);

    struct ActivationBodies
    {
        const char* c;
        const char* javascript;
        const char* python;
        const char* php;
    };

    static const vector<pair<ActivationFunction, ActivationBodies>>& activation_table();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
