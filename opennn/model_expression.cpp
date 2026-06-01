//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "model_expression.h"
#include "scaling_layer.h"
#include "string_utilities.h"
#include "neural_network.h"
#include "variable.h"

namespace opennn
{

namespace {

constexpr const char* c_header =
    "// Artificial Intelligence Techniques SL\n"
    "// artelnics@artelnics.com\n"
    "//\n"
    "// Model exported to C. Edit inputs[] in main() and call calculate_outputs().\n"
    "//\n"
    "// Input names:";

constexpr const char* php_header =
    "<!DOCTYPE html> \n"
    "<!--\n"
    "Artificial Intelligence Techniques SL\n"
    "artelnics@artelnics.com\n\n"
    "Model exported to PHP API. Pass values via URL (e.g. ?num0=...&num1=...).\n\n"
    "\tInput Names: ";

constexpr const char* php_subheader = R"HTML(
-->


<html lang = "en">

<head>

<title>Rest API Client Side Demo</title>

<meta charset = "utf-8">
<meta name = "viewport" content = "width=device-width, initial-scale=1">
<link rel = "stylesheet" href = "https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
<script source = "https://ajax.googleapis.com/ajax/libs/jquery/3.2.0/jquery.min.js"></script>
<script source = "https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
</head>
<style>
.btn{
	background-color: #7393B3;
	border: none;
	color: white;
	padding: 15px 32px;
	text-align: center;
	font-size: 16px;
}
</style>
<body>
<div class = "container">
<br></br><div class = "form-group">
<p>follow the steps defined in the "index.php" file</p>
<p>Refresh the page to see the prediction</p>
</div>
<h4>
<?php

)HTML";

constexpr const char* javascript_header =
    "<!--\n"
    "Artificial Intelligence Techniques SL\n"
    "artelnics@artelnics.com\n\n"
    "Model exported to JavaScript. Form sliders set inputs; click the button to run neuralNetwork().\n\n"
    "Input Names:";

constexpr const char* javascript_subheader = R"HTML(-->

<!DOCTYPE HTML>
<html lang="en">
<head>
<link href="https://www.neuraldesigner.com/assets/css/neuraldesigner.css" rel="stylesheet" />
<link href="https://www.neuraldesigner.com/images/fav.ico" rel="shortcut icon" type="image/x-icon" />
</head>

<style>
body {
display: flex;
justify-content: center;
align-items: center;
min-height: 100vh;
margin: 0;
padding: 2em 0;
background-color: #f0f0f0;
font-family: Arial, sans-serif;
box-sizing: border-box;
}

.content-wrapper {
width: 100%;
max-width: 800px;
margin: 0 auto;
padding: 20px;
background-color: #fff;
box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
border: 1px solid #777;
border-radius: 5px;
text-align: center;
}

.form-table {
border-collapse: collapse;
width: 100%;
margin-bottom: 20px;
border: 1px solid #808080;
}

.form-table th,
.form-table td {
padding: 10px;
text-align: left;
vertical-align: middle;
font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}

.form-table tr:not(:last-child) {
border-bottom: 1px solid #808080;
}

.neural-cell {
text-align: right;
width: 50%;
}

.neural-cell input[float="range"] {
display: block;
margin-left: auto;
margin-right: 0;
box-sizing: border-box;
max-width: 200px;
width: 90%;
}

.neural-cell input[float="number"],
.neural-cell input[float="text"],
.neural-cell select {
display: block;
margin-left: auto;
margin-right: 0;
box-sizing: border-box;
max-width: 200px;
width: 90%;
padding: 5px;
text-align: right;
text-align-last: right;
}

.neural-cell input[float="number"] {
margin-top: 8px;
}

.btn {
background-color: #5da9e9;
border: none;
color: white;
text-align: center;
font-size: 16px;
margin-top: 10px;
margin-bottom: 20px;
cursor: pointer;
padding: 10px 20px;
border-radius: 5px;
transition: background-color 0.3s ease;
font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}

.btn:hover {
background-color: #4b92d3;
}

input[float="range"]::-webkit-slider-runnable-track {
background: #8fc4f0;
height: 0.5rem;
}

input[float="range"]::-moz-range-track {
background: #8fc4f0;
height: 0.5rem;
}

input[float="range"]::-webkit-slider-thumb {
-webkit-appearance: none;
appearance: none;
margin-top: -5px;
background-color: #5da9e9;
border-radius: 50%;
height: 20px;
width: 20px;
border: 2px solid #000000;
box-shadow: 0 0 5px rgba(0,0,0,0.25);
cursor: pointer;
}

input[float="range"]::-moz-range-thumb {
background-color: #5da9e9;
border-radius: 50%;
margin-top: -5px;
height: 20px;
width: 20px;
border: 2px solid #000000;
box-shadow: 0 0 5px rgba(0,0,0,0.25);
cursor: pointer;
}

.form-table th {
background-color: #f2f2f2;
font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}

h4 {
font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
color: #333;
}
</style>

<body>

<section>
<div class="content-wrapper">
<form onsubmit="neuralNetwork(); return false;">
<h4>INPUTS</h4>
<table class="form-table">
)HTML";

constexpr const char* python_header =
    "''' \n"
    "Artificial Intelligence Techniques SL\n"
    "artelnics@artelnics.com\n\n"
    "Model exported to Python. Use NeuralNetwork().calculate_outputs([...]).\n\n"
    "Input Names: \n";

constexpr const char* python_subheader =
    "\nFor batch prediction (input must be np.ndarray):\n"
    "\tnn.calculate_batch_output(np.array([[1, 2], [4, 5]]))\n"
    "''' \n";

}

ModelExpression::ModelExpression(const NeuralNetwork* neural_network) : neural_network(neural_network) {}


vector<string> ModelExpression::split_expression_lines(const string& expression)
{
    vector<string> lines;
    stringstream ss(expression);
    string line;

    while (getline(ss, line, '\n'))
    {
        if (line.empty() || ranges::all_of(line, [](char c) { return isspace(static_cast<unsigned char>(c)); }))
            continue;
        if (line.find("{") != string::npos)
            break;
        if (line.back() != ';')
            line += ';';
        lines.push_back(line);
    }

    return lines;
}

string ModelExpression::get_layer_expression(const Layer& layer,
                                             const vector<string>& input_names,
                                             const vector<string>& output_names)
{
    return layer.write_expression(input_names, output_names);
}

string ModelExpression::build_expression() const
{
    const size_t layers_number = neural_network->get_layers_number();
    const vector<string> layer_labels = neural_network->get_layer_labels();

    const Index inputs_number = neural_network->get_inputs_number();
    const Index outputs_number = neural_network->get_outputs_number();

    vector<string> new_input_names = neural_network->get_input_feature_names();
    new_input_names.resize(inputs_number);
    for (Index i = 0; i < inputs_number; ++i)
        if (new_input_names[i].empty())
            new_input_names[i] = format("input_{}", i);

    vector<string> new_output_names = neural_network->get_output_feature_names();
    new_output_names.resize(outputs_number);
    for (Index i = 0; i < outputs_number; ++i)
        if (new_output_names[i].empty())
            new_output_names[i] = format("output_{}", i);

    ostringstream buffer;

    const auto& layers = neural_network->get_layers();

    for (size_t i = 0; i < layers_number; ++i)
    {
        const bool is_last = (i == layers_number - 1);
        vector<string> layer_output_names;

        if (is_last)
        {
            layer_output_names = new_output_names;
        }
        else
        {
            const Index layer_neurons_number = layers[i]->get_outputs_number();
            layer_output_names.resize(layer_neurons_number);
            for (size_t j = 0; j < static_cast<size_t>(layer_neurons_number); ++j)
                layer_output_names[j] = (layer_labels[i] == "scaling_layer" && j < new_input_names.size())
                                            ? "scaled_" + new_input_names[j]
                                            : format("{}_output_{}", layer_labels[i], j);
        }

        buffer << get_layer_expression(*layers[i], new_input_names, layer_output_names) << "\n";

        if (!is_last)
            new_input_names = move(layer_output_names);
    }

    return buffer.str();
}

void ModelExpression::apply_name_mapping(string& text, const vector<string>& original, const vector<string>& mapped)
{
    const size_t count = min(original.size(), mapped.size());
    for (size_t i = 0; i < count; ++i)
        replace_all_word_appearances(text, original[i], mapped[i]);
}

string ModelExpression::process_body_line(const string& line, const vector<string>& input_names, const vector<string>& fixed_input_names)
{
    string processed = line;
    replace_all_appearances(processed, "[", "_");
    replace_all_appearances(processed, "]", "_");
    apply_name_mapping(processed, input_names, fixed_input_names);
    return processed;
}

vector<string> ModelExpression::prepare_body_lines(const string& expression)
{
    vector<string> lines = split_expression_lines(expression);
    rename_spaced_var_definitions(lines);
    return lines;
}

void ModelExpression::rename_spaced_var_definitions(vector<string>& lines)
{
    for (size_t i = 0; i < lines.size(); ++i)
    {
        const size_t equal_pos = lines[i].find('=');
        if (equal_pos == string::npos) continue;

        const string var_def = lines[i].substr(0, equal_pos);

        const size_t first = var_def.find_first_not_of(" \t");
        if (first == string::npos) continue;

        const size_t last = var_def.find_last_not_of(" \t");
        const string clean_var = var_def.substr(first, (last - first + 1));

        if (clean_var.find(' ') == string::npos) continue;

        string fixed_var = clean_var;
        ranges::replace(fixed_var, ' ', '_');

        for (size_t j = 0; j < lines.size(); ++j)
            replace_all_appearances(lines[j], clean_var, fixed_var);
    }
}

const vector<pair<string, ModelExpression::ActivationBodies>>& ModelExpression::activation_table()
{
    static const vector<pair<string, ActivationBodies>> table = {
        {"Identity", {
            "float Identity (float x) {\n\treturn x;\n}\n\n",
            "\nfunction Identity(x) {\n\treturn x;\n}\n",
            "\t@staticmethod\n\tdef Identity(x):\n\t\treturn x\n\n",
            "function Identity($x) { return $x; }\n"
        }},
        {"Sigmoid", {
            "float Sigmoid(float x) {\n\tfloat z = 1.0f / (1.0f + expf(-x));\n\treturn z;\n}\n\n",
            "function Sigmoid(x) {\n\tvar z = 1/(1+Math.exp(-x));\n\treturn z;\n}\n",
            "\t@staticmethod\n\tdef Sigmoid (x):\n\t\tz = 1/(1+np.exp(-x))\n\t\treturn z\n\n",
            "function Sigmoid($x) { return 1 / (1 + exp(-$x)); }\n"
        }},
        {"ReLU", {
            "float ReLU(float x) {\n\tfloat z = fmaxf(0.0f, x);\n\treturn z;\n}\n\n",
            "function ReLU(x) {\n\tvar z = Math.max(0, x);\n\treturn z;\n}\n",
            "\t@staticmethod\n\tdef ReLU (x):\n\t\tz = np.maximum(0, x)\n\t\treturn z\n\n",
            "function ReLU($x) { return max(0, $x); }\n"
        }},
        {"ExponentialLinear", {
            "float ExponentialLinear(float x) {\n\tfloat z;\n\tconst float alpha = 1.67326f;\n\tif (x > 0.0f) {\n\t\tz = x;\n\t} else {\n\t\tz = alpha * (expf(x) - 1.0f);\n\t}\n\treturn z;\n}\n\n",
            "function ExponentialLinear(x) {\n\tvar alpha = 1.67326;\n\tif(x>0) {\n\t\tvar z = x;\n\t} else {\n\t\tvar z = alpha*(Math.exp(x)-1);\n\t}\n\treturn z;\n}\n",
            "\t@staticmethod\n\tdef ExponentialLinear (x):\n\t\talpha = 1.67326\n\t\treturn np.where(x > 0, x, alpha * (np.exp(x) - 1))\n\n",
            "function ExponentialLinear($x) { $alpha = 1.67326; return ($x > 0) ? $x : $alpha * (exp($x) - 1); }\n"
        }},
        {"Tanh", {
            "float Tanh(float x) {\n\treturn tanhf(x);\n}\n\n",
            "function Tanh(x) {\n\treturn Math.tanh(x);\n}\n",
            "\t@staticmethod\n\tdef Tanh(x):\n\t\treturn np.tanh(x)\n\n",
            "function Tanh($x) { return tanh($x); }\n"
        }},
        {"Softmax", {
            "float Softmax(float x) {\n\treturn expf(x);\n}\n\n",
            "function Softmax(x) {\n\treturn Math.exp(x);\n}\n",
            "\t@staticmethod\n\tdef Softmax(x):\n\t\treturn np.exp(x)\n\n",
            "function Softmax($x) { return exp($x); }\n"
        }}
    };
    return table;
}

string ModelExpression::get_expression_c() const
{
    const vector<string> output_names = neural_network->get_output_feature_names();

    string expression = build_expression();
    apply_name_mapping(expression, output_names, fix_names(output_names, "output_"));
    const vector<string> lines = prepare_body_lines(expression);
    const bool has_softmax = expression.find("Softmax") != string::npos;

    ostringstream buffer;
    emit_c_prelude(buffer);
    emit_c_activations(buffer, expression);
    emit_c_calculate_outputs(buffer, expression, lines, has_softmax);
    emit_c_main(buffer);
    return buffer.str();
}

void ModelExpression::emit_c_prelude(ostringstream& buffer) const
{
    const vector<string> input_names = neural_network->get_input_feature_names();

    buffer << c_header;

    for (size_t i = 0; i < input_names.size(); ++i)
        buffer << "\n// \t " << i << ")  " << input_names[i];

    buffer << "\n \n \n#include <stdio.h>\n"
              "#include <stdlib.h>\n"
              "#include <math.h>\n\n";
}

void ModelExpression::emit_c_activations(ostringstream& buffer, const string& expression) const
{
    for (const auto& [name, bodies] : activation_table())
        if (name == "Identity" || expression.find(name) != string::npos)
            buffer << bodies.c;
}

void ModelExpression::emit_c_calculate_outputs(ostringstream& buffer,
                                               const string& expression,
                                               const vector<string>& lines,
                                               bool has_softmax) const
{
    const vector<string> input_names = neural_network->get_input_feature_names();
    const vector<string> output_names = neural_network->get_output_feature_names();
    const vector<string> fixed_input_names = fix_names(input_names, "input_");
    const vector<string> fixed_output_names = fix_names(output_names, "output_");

    const Index inputs_number = input_names.size();
    const Index outputs_number = output_names.size();

    buffer << "float* calculate_outputs(const float* inputs)\n{\n";

    for (Index i = 0; i < inputs_number; ++i)
        buffer << "\tconst float " << fixed_input_names[i] << " = inputs[" << i << "];\n";

    buffer << "\n";

    for (const string& l : lines)
        buffer << "\tdouble " << process_body_line(l, input_names, fixed_input_names) << "\n";

    const vector<string> fixed_outputs = fix_output_names(expression, output_names, ProgrammingLanguage::C);
    if (!fixed_outputs.empty())
    {
        buffer << "\n";
        for (const string& l : fixed_outputs)
            buffer << "\t" << l << "\n";
    }

    buffer << "\n\tfloat* out = (float*)malloc(" << outputs_number << " * sizeof(float));\n"
              "\tif (out == NULL) {\n"
              "\t\tprintf(\"Error: Memory allocation failed in calculate_outputs.\\n\");\n"
              "\t\treturn NULL;\n"
              "\t}\n\n";

    for (Index i = 0; i < outputs_number; ++i)
        buffer << "\tout[" << i << "] = " << fixed_output_names[i] << ";\n";

    if (has_softmax)
        buffer << "\n\t// Softmax Normalization\n"
                  "\tfloat sum = 0.0f;\n"
                  "\tfor(int i = 0; i < " << outputs_number << "; ++i) sum += out[i];\n"
                  "\tfor(int i = 0; i < " << outputs_number << "; ++i) out[i] /= sum;\n";

    buffer << "\n\treturn out;\n}\n\n";
}

void ModelExpression::emit_c_main(ostringstream& buffer) const
{
    const vector<string> input_names = neural_network->get_input_feature_names();
    const vector<string> output_names = neural_network->get_output_feature_names();

    const Index inputs_number = input_names.size();
    const Index outputs_number = output_names.size();

    buffer << "int main() { \n\n"
              "\tfloat* inputs = (float*)malloc(" << inputs_number << " * sizeof(float)); \n"
              "\tif (inputs == NULL) {\n"
              "\t\tprintf(\"Error: Memory allocation failed for inputs.\\n\");\n"
              "\t\treturn 1;\n"
              "\t}\n\n"
              "\t// Please enter your values here:\n";

    for (Index i = 0; i < inputs_number; ++i)
        buffer << "\tinputs[" << i << "] = 0.0f; // " << input_names[i] << "\n";

    buffer << "\n\tfloat* outputs;\n"
              "\n\toutputs = calculate_outputs(inputs);\n\n"
              "\tif (outputs != NULL) {\n"
              "\t\tprintf(\"These are your outputs:\\n\");\n";

    for (Index i = 0; i < outputs_number; ++i)
        buffer << "\t\tprintf(\""<< output_names[i] << ": %f \\n\", outputs[" << i << "]);\n";

    buffer << "\t}\n\n"
              "\t// Free the allocated memory\n"
              "\tfree(inputs);\n"
              "\tfree(outputs);\n\n"
              "\treturn 0;\n} \n";
}

string ModelExpression::get_expression_php() const
{
    const vector<string> input_names = neural_network->get_input_feature_names();
    const vector<string> output_names = neural_network->get_output_feature_names();
    const vector<string> fixed_input_names = fix_names(input_names, "input_");
    const vector<string> fixed_output_names = fix_names(output_names, "output_");

    string expression = build_expression();

    vector<string> all_possible_vars = fixed_output_names;
    all_possible_vars.insert(all_possible_vars.end(), fixed_input_names.begin(), fixed_input_names.end());
    ranges::sort(all_possible_vars, [](const string& a, const string& b) { return a.length() > b.length(); });
    for (const string& var_name : all_possible_vars)
        replace_all_word_appearances(expression, var_name, "$" + var_name);

    for (size_t i = 0; i < input_names.size(); ++i)
        if (input_names[i] != fixed_input_names[i])
            replace_all_word_appearances(expression, input_names[i], "$" + fixed_input_names[i]);

    for (size_t i = 0; i < output_names.size(); ++i)
        if (output_names[i] != fixed_output_names[i])
            replace_all_word_appearances(expression, output_names[i], "$" + fixed_output_names[i]);

    vector<string> lines = split_expression_lines(expression);
    for (string& l : lines)
    {
        replace_all_appearances(l, "[", "_");
        replace_all_appearances(l, "]", "_");
    }
    rename_spaced_var_definitions(lines);

    const bool has_softmax = expression.find("Softmax") != string::npos;

    ostringstream buffer;
    emit_php_prelude(buffer);
    emit_php_activations(buffer, expression);
    emit_php_inputs_setup(buffer);
    emit_php_body(buffer, lines, has_softmax);
    emit_php_response(buffer);
    return buffer.str();
}

void ModelExpression::emit_php_prelude(ostringstream& buffer) const
{
    const vector<string> input_names = neural_network->get_input_feature_names();

    buffer << php_header;
    for (size_t i = 0; i < input_names.size(); ++i)
        buffer << "\n\t\t" << i << ")  " << input_names[i];
    buffer << php_subheader;
}

void ModelExpression::emit_php_activations(ostringstream& buffer, const string& expression) const
{
    for (const auto& [name, bodies] : activation_table())
        if (expression.find(name) != string::npos)
            buffer << bodies.php;
}

void ModelExpression::emit_php_inputs_setup(ostringstream& buffer) const
{
    const vector<string> fixed_input_names = fix_names(neural_network->get_input_feature_names(), "input_");

    buffer << "\nsession_start();\n"
              "if (isset($_GET['num0'])) { \n"
              "$params = $_GET;\n\n";

    for (size_t i = 0; i < fixed_input_names.size(); ++i)
        buffer << "$" << fixed_input_names[i] << " = isset($params['num" << i << "']) ? floatval($params['num" << i << "']) : 0;\n";

    buffer << "\n";
}

void ModelExpression::emit_php_body(ostringstream& buffer, const vector<string>& lines, bool has_softmax) const
{
    const vector<string> fixed_output_names = fix_names(neural_network->get_output_feature_names(), "output_");

    for (const string& l : lines)
        buffer << l << "\n";

    if (has_softmax)
    {
        buffer << "\n// Softmax Normalization\n$sum = 0;\n";
        for (size_t i = 0; i < fixed_output_names.size(); ++i)
            buffer << "$sum += $" << fixed_output_names[i] << ";\n";
        for (size_t i = 0; i < fixed_output_names.size(); ++i)
            buffer << "$" << fixed_output_names[i] << " /= $sum;\n";
    }
}

void ModelExpression::emit_php_response(ostringstream& buffer) const
{
    const vector<string> output_names = neural_network->get_output_feature_names();
    const vector<string> fixed_output_names = fix_names(output_names, "output_");

    buffer << "\n$response = ['status' => 200,  'status_message' => 'ok'";

    for (size_t i = 0; i < output_names.size(); ++i)
        buffer << ", '" << output_names[i] << "' => $" << fixed_output_names[i];

    buffer << "];\n\n"
              "$json_response_pretty = json_encode($response, JSON_PRETTY_PRINT);\n"
              "echo nl2br(\"\\n\" . $json_response_pretty . \"\\n\");\n"
              "} else {\n echo \"Please provide input values in the URL (e.g., ?num0=value0&num1=value1...)\"; \n}\n"
              "$_SESSION['lastpage'] = __FILE__;\n"
              "?>\n</h4>\n</div>\n</body>\n</html>";
}

string ModelExpression::get_expression_javascript() const
{
    const vector<string> output_names = neural_network->get_output_feature_names();
    const vector<string> fixes_output_names = fix_names(output_names, "output_");

    string expression = build_expression();
    apply_name_mapping(expression, output_names, fixes_output_names);
    replace_all_appearances(expression, "[", "_");
    replace_all_appearances(expression, "]", "_");

    vector<string> lines;
    stringstream ss(expression);
    string token;
    while (getline(ss, token, '\n'))
    {
        if (token.empty()) continue;
        if (token.size() > 1 && token.back() == '{') break;
        if (token.size() > 1 && token.back() != ';') token += ';';
        lines.push_back(token);
    }
    rename_spaced_var_definitions(lines);

    const bool has_softmax = expression.find("Softmax") != string::npos;
    const bool use_category_select = output_names.size() > 5;

    ostringstream buffer;
    emit_js_prelude(buffer);
    emit_js_inputs_html(buffer);
    emit_js_outputs_html(buffer, use_category_select);
    emit_js_runtime(buffer, expression, lines, has_softmax, use_category_select);
    return buffer.str();
}

void ModelExpression::emit_js_prelude(ostringstream& buffer) const
{
    const vector<string> input_names = neural_network->get_input_feature_names();

    buffer << javascript_header;
    for (size_t i = 0; i < input_names.size(); ++i)
        buffer << "\n\t " << i + 1 << ")  " << input_names[i];
    buffer << javascript_subheader;
}

void ModelExpression::emit_js_inputs_html(ostringstream& buffer) const
{
    const vector<string> input_names = neural_network->get_input_feature_names();
    const vector<string> fixes_feature_names = fix_names(input_names, "input_");

    const Scaling* scaling_layer = neural_network->has(LayerType::Scaling)
        ? static_cast<const Scaling*>(neural_network->get_first(LayerType::Scaling))
        : nullptr;
    const VectorR scaling_minimums = scaling_layer ? scaling_layer->get_minimums() : VectorR();
    const VectorR scaling_maximums = scaling_layer ? scaling_layer->get_maximums() : VectorR();
    const bool has_scaling = scaling_layer;

    const Index inputs_number = input_names.size();

    for (Index i = 0; i < inputs_number; ++i)
    {
        float min_value = -1.0f;
        float max_value = 1.0f;
        if (has_scaling
           && i < ssize(scaling_minimums)
           && i < ssize(scaling_maximums))
        {
            min_value = scaling_minimums[i];
            max_value = scaling_maximums[i];
        }

        const float initial_value = (min_value == 0 && max_value == 0)
                                        ? min_value
                                        : (min_value + max_value) / 2;
        const float step = (max_value - min_value) / 100;
        const string& id = fixes_feature_names[i];
        const char* marker = has_scaling ? "scaling layer" : "no scaling layer";

        buffer << "<!-- "<< to_string(i) << marker << " -->\n"
               << "<tr style=\"height:3.5em\">\n"
               << "<td> " << input_names[i] << " </td>\n"
               << "<td class=\"neural-cell\">\n"
               << "<input type=\"range\" id=\"" << id << "\" value=\"" << initial_value << "\" min=\"" << min_value << "\" max=\"" << max_value << "\" step=\"" << step << "\" onchange=\"updateTextInput1(this.value, '" << id << "_text')\" />\n"
               << "<input" << (has_scaling ? " class=\"tabla\"" : "") << " type=\"number\" id=\"" << id << "_text\" value=\"" << initial_value << "\" min=\"" << min_value << "\" max=\"" << max_value << "\" step=\"any\" onchange=\"updateTextInput1(this.value, '" << id << "')\">\n"
               << "</td>\n"
               << "</tr>\n\n";
    }

    buffer << "</table>\n";
}

void ModelExpression::emit_js_outputs_html(ostringstream& buffer, bool use_category_select) const
{
    const vector<string> output_names = neural_network->get_output_feature_names();
    const vector<string> fixes_output_names = fix_names(output_names, "output_");

    const Index outputs_number = output_names.size();

    if (use_category_select)
    {
        buffer << "<!-- HIDDEN INPUTS -->\n";
        for (Index i = 0; i < outputs_number; ++i)
            buffer << "<input type=\"hidden\" id=\"" << fixes_output_names[i] << "\" value=\"\">\n";
        buffer << "\n\n";
    }

    buffer << "<!-- BUTTON HERE -->\n"
           << "<button type=\"button\" class=\"btn\" onclick=\"neuralNetwork()\">calculate outputs</button>\n"
           << "<br/>\n\n"
           << "<table border=\"1px\" class=\"form-table\">\n"
           << "<h4> OUTPUTS </h4>\n";

    if (use_category_select)
    {
        buffer << "<tr style=\"height:3.5em\">\n"
               << "<td> Target </td>\n"
               << "<td class=\"neural-cell\">\n"
               << "<select id=\"category_select\" onchange=\"updateSelectedCategory()\">\n";
        for (Index i = 0; i < outputs_number; ++i)
            buffer << "<option value=\"" << output_names[i] << "\">" << output_names[i] << "</option>\n";
        buffer << "</select>\n"
               << "</td>\n"
               << "</tr>\n\n"
               << "<tr style=\"height:3.5em\">\n"
               << "<td> Value </td>\n"
               << "<td class=\"neural-cell\">\n"
               << "<input style=\"text-align:right; padding-right:20px;\" id=\"selected_value\" value=\"\" type=\"text\"  disabled/>\n"
               << "</td>\n"
               << "</tr>\n\n";
    }
    else
    {
        for (Index i = 0; i < outputs_number; ++i)
            buffer << "<tr style=\"height:3.5em\">\n"
                   << "<td> " << output_names[i] << " </td>\n"
                   << "<td class=\"neural-cell\">\n"
                   << "<input style=\"text-align:right; padding-right:20px;\" id=\"" << fixes_output_names[i] << "\" value=\"\" type=\"text\"  disabled/>\n"
                   << "</td>\n"
                   << "</tr>\n\n";
    }

    buffer << "</table>\n\n</form>\n</div>\n\n</section>\n\n";
}

void ModelExpression::emit_js_runtime(ostringstream& buffer,
                                      const string& expression,
                                      const vector<string>& lines,
                                      bool has_softmax,
                                      bool use_category_select) const
{
    const vector<string> input_names = neural_network->get_input_feature_names();
    const vector<string> output_names = neural_network->get_output_feature_names();
    const vector<string> fixes_feature_names = fix_names(input_names, "input_");
    const vector<string> fixes_output_names = fix_names(output_names, "output_");

    const Index inputs_number = input_names.size();
    const Index outputs_number = output_names.size();

    buffer << "<script>\n";

    if (use_category_select)
    {
        buffer << "function updateSelectedCategory() {\n"
               << "\tvar selectedCategory = document.getElementById(\"category_select\").value;\n"
               << "\tvar selectedValueElement = document.getElementById(\"selected_value\");\n";
        for (Index i = 0; i < outputs_number; ++i)
            buffer << "\tif(selectedCategory === \"" << fixes_output_names[i] << "\") {\n"
                   << "\t\tselectedValueElement.value = document.getElementById(\"" << fixes_output_names[i] << "\").value;\n"
                   << "\t}\n";
        buffer << "}\n\n";
    }

    for (const auto& [name, bodies] : activation_table())
        if (name == "Identity" || name == "Tanh" || expression.find(name) != string::npos)
            buffer << bodies.javascript;
    buffer << "\n";

    buffer << "function neuralNetwork()\n{\n\tvar inputs = [];\n";
    for (Index i = 0; i < inputs_number; ++i)
    {
        buffer << "\tvar " << fixes_feature_names[i] << " = (parseFloat(document.getElementById(\"" << fixes_feature_names[i] << "_text\").value) || 0); \n";
        buffer << "\tinputs.push(" << fixes_feature_names[i] << ");\n";
    }
    buffer << "\n\tvar outputs = calculate_outputs(inputs); \n";

    if (use_category_select)
        buffer << "\tupdateSelectedCategory();\n";
    else
        for (Index i = 0; i < outputs_number; ++i)
            buffer << "\tvar " << fixes_output_names[i] << " = document.getElementById(\"" << fixes_output_names[i] << "\");\n"
                   << "\t" << fixes_output_names[i] << ".value = outputs[" << to_string(i) << "].toFixed(4);\n";

    buffer << "}\nfunction calculate_outputs(inputs)\n{\n";
    for (Index i = 0; i < inputs_number; ++i)
        buffer << "\tvar " << fixes_feature_names[i] << " = +inputs[" << to_string(i) << "];\n";
    buffer << "\n";

    static const char* const math_keywords[] = {"exp", "tanh", "max", "min"};

    for (const string& raw_line : lines)
    {
        string line = raw_line;
        for (Index j = 0; j < inputs_number; ++j)
            replace_all_word_appearances(line, input_names[j], fixes_feature_names[j]);

        if (has_softmax) replace_all_appearances(line, "Softmax", "___SOFTMAX_TOKEN___");
        for (const char* kw : math_keywords)
            replace_all_appearances(line, kw, string("Math.") + kw);
        if (has_softmax) replace_all_appearances(line, "___SOFTMAX_TOKEN___", "Softmax");

        replace_all_appearances(line, "nan", "0");
        replace_all_appearances(line, "NaN", "0");
        replace_all_appearances(line, "inf", "Infinity");

        if (line.size() <= 1) buffer << "\n";
        else                 buffer << "\tvar " << line << "\n";
    }

    const vector<string> fixed_outputs = fix_output_names(expression, output_names, ProgrammingLanguage::JavaScript);
    for (const string& l : fixed_outputs)
        buffer << l << "\n";

    buffer << "\tvar out = [];\n";
    for (Index i = 0; i < outputs_number; ++i)
        buffer << "\tout.push(" << fixes_output_names[i] << ");\n";

    if (has_softmax)
        buffer << "\n\t// Softmax Normalization\n"
                  "\tvar sum = 0;\n"
                  "\tfor(var i = 0; i < out.length; ++i) sum += out[i];\n"
                  "\tfor(var i = 0; i < out.length; ++i) out[i] /= sum;\n";

    buffer << "\n\treturn out;\n}\n\n"
              "function updateTextInput1(value, id)\n{\n"
              "\tdocument.getElementById(id).value = value;\n"
              "}\n\n"
              "</script>\n\n"
              "<!--script source=\"https://www.neuraldesigner.com/app/htmlparts/footer.js\"></script-->\n\n"
              "</body>\n\n"
              "</html>\n";
}

string ModelExpression::get_expression_python() const
{
    string expression = build_expression();
    for (const string& name : neural_network->get_output_feature_names())
        replace_all_word_appearances(expression, name, replace_reserved_keywords(name));

    const vector<string> lines = prepare_body_lines(expression);
    const bool has_softmax = expression.find("Softmax") != string::npos;

    ostringstream buffer;
    emit_python_prelude(buffer);
    emit_python_class_header(buffer);
    emit_python_activations(buffer, expression);
    emit_python_calculate_outputs(buffer, lines, has_softmax);
    emit_python_batch_and_main(buffer);

    string out = buffer.str();
    replace(out, ";", "");
    return out;
}

void ModelExpression::emit_python_prelude(ostringstream& buffer) const
{
    const vector<string> input_names = neural_network->get_input_feature_names();

    buffer << python_header;
    for (size_t i = 0; i < input_names.size(); ++i)
        buffer << "\t" << i << ") " << input_names[i] << "\n";
    buffer << python_subheader;
}

void ModelExpression::emit_python_class_header(ostringstream& buffer) const
{
    const vector<string> input_names = neural_network->get_input_feature_names();

    buffer << "import numpy as np\n"
              "import pandas as pd\n\n"
              "class NeuralNetwork:\n\n";

    string inputs_list_str;
    for (size_t i = 0; i < input_names.size(); ++i)
    {
        if (i) inputs_list_str += ", ";
        inputs_list_str += "'" + replace_reserved_keywords(input_names[i]) + "'";
    }

    buffer << "\tdef __init__(self):\n"
           << "\t\tself.inputs_number = " << input_names.size() << "\n"
           << "\t\tself.input_names = [" << inputs_list_str << "]\n\n";
}

void ModelExpression::emit_python_activations(ostringstream& buffer, const string& expression) const
{
    for (const auto& [name, bodies] : activation_table())
        if (name == "Identity" || expression.find(name) != string::npos)
            buffer << bodies.python;
}

void ModelExpression::emit_python_calculate_outputs(ostringstream& buffer,
                                                    const vector<string>& lines,
                                                    bool has_softmax) const
{
    const vector<string> input_names = neural_network->get_input_feature_names();
    const vector<string> fixed_output_names = fix_names(neural_network->get_output_feature_names(), "output_");

    buffer << "\tdef calculate_outputs(self, inputs):\n";

    vector<string> python_mapped(input_names.size());
    ranges::transform(input_names, python_mapped.begin(), replace_reserved_keywords);

    for (size_t i = 0; i < input_names.size(); ++i)
        buffer << "\t\t" << python_mapped[i] << " = inputs[" << i << "]\n";

    buffer << "\n";

    for (const string& l : lines)
    {
        string processed_line = process_body_line(l, input_names, python_mapped);

        for (const auto& [name, _] : activation_table())
            replace_all_word_appearances(processed_line, name, "self." + name);

        replace(processed_line, ";", "");
        buffer << "\t\t" << processed_line << "\n";
    }

    string return_list = "[";
    for (size_t i = 0; i < fixed_output_names.size(); ++i)
    {
        if (i) return_list += ", ";
        return_list += fixed_output_names[i];
    }
    return_list += "]";

    buffer << "\t\toutputs = " << return_list << "\n";
    if (has_softmax)
    {
        buffer << "\t\tsum_val = np.sum(outputs)\n";
        buffer << "\t\toutputs = [x / sum_val for x in outputs]\n";
    }
    buffer << "\t\treturn outputs\n\n";
}

void ModelExpression::emit_python_batch_and_main(ostringstream& buffer) const
{
    const vector<string> input_names = neural_network->get_input_feature_names();
    const vector<string> fixed_output_names = fix_names(neural_network->get_output_feature_names(), "output_");

    const Index inputs_number = input_names.size();
    const Index outputs_number = fixed_output_names.size();

    buffer << "\tdef calculate_batch_output(self, input_batch):\n"
              "\t\toutput_batch = np.zeros((len(input_batch), " << to_string(outputs_number) << "))\n"
              "\t\tfor i in range(len(input_batch)):\n"
              "\t\t\tinputs = list(input_batch[i])\n"
              "\t\t\toutput = self.calculate_outputs(inputs)\n"
              "\t\t\toutput_batch[i] = output\n"
              "\t\treturn output_batch\n\n";

    buffer << "def main():\n"
              "\n\t# Introduce your input values here\n";

    const vector<string> fixed_variable_names = fix_names(input_names, "input_");

    for (Index i = 0; i < inputs_number; ++i)
        buffer << "\t" << fixed_variable_names[i] << " = 0  # " << input_names[i] << "\n";

    buffer << "\n\t# --- Data conversion (DO NOT modify) ---\n"
              "\tinputs = []\n\n";

    for (Index i = 0; i < inputs_number; ++i)
        buffer << "\tinputs.append(" << fixed_variable_names[i] << ")\n";

    buffer << "\n\tnn = NeuralNetwork()\n"
              "\toutputs = nn.calculate_outputs(inputs)\n"
              "\tprint(outputs)\n\n"
              "if __name__ == \"__main__\":\n"
              "\tmain()\n";
}

string ModelExpression::replace_reserved_keywords(const string& input)
{
    static const unordered_map<char, string_view> char_replacements = {
        {' ', "_"},        {'.', "_dot_"},    {'/', "_div_"},    {'*', "_mul_"},
        {'+', "_sum_"},    {'-', "_res_"},    {'=', "_equ_"},    {'!', "_not_"},
        {',', "_colon_"},  {';', "_semic_"},  {'\\', "_slash_"}, {'&', "_amprsn_"},
        {'?', "_ntrgtn_"}, {'<', "_lower_"},  {'>', "_higher_"}
    };

    static const unordered_map<string, string> special_words = {
        {"min", "mi_n"}, {"max", "ma_x"}, {"exp", "ex_p"}, {"tanh", "ta_nh"}
    };

    string out;

    if (input[0] == '$')
        out = input;

    for (const char character : input)
    {
        const auto it = char_replacements.find(character);
        if (it != char_replacements.end())
            out += it->second;
        else if (isalnum(static_cast<unsigned char>(character)) || character == '_')
            out += character;
    }

    if (!out.empty() && isdigit(static_cast<unsigned char>(out[0])))
        out = '_' + out;

    for (const auto& [search, replace_val] : special_words)
        replace_all_appearances(out, search, replace_val);

    return out;
}

vector<string> ModelExpression::fix_output_names(const string& str,
                                                           const vector<string>& outputs,
                                                           const ProgrammingLanguage& programming_language)
{
    vector<string> out;
    vector<string> tokens;

    string token;
    stringstream ss(str);

    const size_t num_outputs = outputs.size();

    while (getline(ss, token, '\n'))
    {
        if (token.empty() || ranges::all_of(token, [](char c) { return isspace(c); }))
            continue;

        if (token.find("{") != string::npos)
            break;

        if (token.find("=") != string::npos)
            tokens.push_back(token);
    }

    if (tokens.size() < num_outputs)
        return {};

    const vector<string> fixed_outputs = fix_names(outputs, "output_");

    struct DeclFormat { const char* lhs_prefix; const char* rhs_prefix; const char* suffix; };
    static const unordered_map<ProgrammingLanguage, DeclFormat> formats =
    {
        {ProgrammingLanguage::C,          {"double ", "",  ";"}},
        {ProgrammingLanguage::JavaScript, {"\tvar ",  "",  ";"}},
        {ProgrammingLanguage::Python,     {"",        "",  ""}},
        {ProgrammingLanguage::PHP,        {"$",       "$", ";"}}
    };

    const DeclFormat& fmt = formats.at(programming_language);
    const size_t tokens_count = tokens.size();

    for (size_t i = 0; i < num_outputs; ++i)
    {
        const string intermediate_var_name = get_first_word(tokens[tokens_count - num_outputs + i]);
        const string& final_output_name = fixed_outputs[i];

        if (final_output_name != intermediate_var_name)
            out.push_back(fmt.lhs_prefix + final_output_name + " = " + fmt.rhs_prefix + intermediate_var_name + fmt.suffix);
    }

    return out;
}

vector<string> ModelExpression::fix_names(const vector<string>& names, const string& default_prefix)
{
    vector<string> fixed(names.size());

    for (size_t i = 0; i < names.size(); ++i)
        fixed[i] = names[i].empty()
            ? format("{}{}", default_prefix, i)
            : replace_reserved_keywords(names[i]);

    return fixed;
}

void ModelExpression::save(const filesystem::path& file_name, ProgrammingLanguage language) const
{
    ofstream file(file_name);

    if (!file.is_open())
        throw runtime_error(format("Cannot open file: {}", file_name.string()));

    using enum ProgrammingLanguage;
    switch (language)
    {
    case C:          file << get_expression_c();          break;
    case Python:     file << get_expression_python();     break;
    case JavaScript: file << get_expression_javascript(); break;
    case PHP:        file << get_expression_php();        break;
    }

    file.close();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
