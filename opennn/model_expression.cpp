//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "model_expression.h"
#include "scaling_layer.h"
#include "unscaling_layer.h"
#include "bounding_layer.h"
#include "dense_layer.h"
#include "recurrent_layer.h"
#include "long_short_term_memory_layer.h"
#include "string_utilities.h"
#include "neural_network.h"
#include "variable.h"

namespace opennn
{

static constexpr const char* c_header =
    "// Artificial Intelligence Techniques SL\n"
    "// artelnics@artelnics.com\n"
    "//\n"
    "// Model exported to C. Edit inputs[] in main() and call calculate_outputs().\n"
    "//\n"
    "// calculate_outputs() uses no heap and returns a pointer to a static buffer\n"
    "// (not thread-safe), so it can be embedded in microcontroller firmware.\n"
    "// Define OPENNN_EXPORT_NO_MAIN to exclude main() when embedding.\n"
    "//\n"
    "// Input names:";

static constexpr const char* php_header =
    "<!DOCTYPE html> \n"
    "<!--\n"
    "Artificial Intelligence Techniques SL\n"
    "artelnics@artelnics.com\n\n"
    "Model exported to PHP API. Pass values via URL (e.g. ?num0=...&num1=...).\n\n"
    "\tInput Names: ";

static constexpr const char* php_subheader = R"HTML(
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

static constexpr const char* javascript_header =
    "<!--\n"
    "Artificial Intelligence Techniques SL\n"
    "artelnics@artelnics.com\n\n"
    "Model exported to JavaScript. Form sliders set inputs; click the button to run neuralNetwork().\n\n"
    "Input Names:";

static constexpr const char* javascript_subheader = R"HTML(-->

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

static constexpr const char* python_header =
    "''' \n"
    "Artificial Intelligence Techniques SL\n"
    "artelnics@artelnics.com\n\n"
    "Model exported to Python. Use NeuralNetwork().calculate_outputs([...]).\n\n"
    "Input Names: \n";

static constexpr const char* python_subheader =
    "\nFor batch prediction (input must be np.ndarray):\n"
    "\tnn.calculate_batch_output(np.array([[1, 2], [4, 5]]))\n"
    "''' \n";

ModelExpression::ModelExpression(const NeuralNetwork* neural_network) : neural_network(neural_network) {}

Index ModelExpression::get_flat_inputs_number() const
{
    const auto& layers = neural_network->get_layers();

    if (!layers.empty())
    {
        const Shape first_input_shape = layers[0]->get_input_shape();
        if (first_input_shape.rank == 2)
            return first_input_shape.size();
    }

    return neural_network->get_inputs_number();
}

vector<string> ModelExpression::get_flat_input_names() const
{
    vector<string> names = neural_network->get_input_feature_names();
    const Index flat_inputs_number = get_flat_inputs_number();

    if (ssize(names) == flat_inputs_number)
        return names;

    const auto& layers = neural_network->get_layers();
    const Shape first_input_shape = layers.empty() ? Shape{} : layers[0]->get_input_shape();

    if (first_input_shape.rank == 2)
    {
        const Index time_steps = first_input_shape[0];
        const Index features = first_input_shape[1];

        vector<string> expanded(static_cast<size_t>(flat_inputs_number));

        for (Index t = 0; t < time_steps; ++t)
            for (Index f = 0; f < features; ++f)
            {
                const string base = (f < ssize(names) && !names[size_t(f)].empty())
                    ? names[size_t(f)]
                    : format("input_{}", f);
                expanded[size_t(t * features + f)] = format("{}_t{}", base, t);
            }

        return expanded;
    }

    names.resize(size_t(flat_inputs_number));
    return names;
}


vector<string> ModelExpression::split_expression_lines(const string& expression)
{
    vector<string> lines;
    stringstream ss(expression);
    string line;

    while (getline(ss, line, '\n'))
    {
        if (line.empty() || ranges::all_of(line, [](char c) { return isspace(static_cast<unsigned char>(c)); }))
            continue;
        if (line.find('{') != string::npos)
            break;
        if (line.back() != ';')
            line += ';';
        lines.push_back(line);
    }

    return lines;
}

void ModelExpression::check_parameters_are_finite() const
{
    if (neural_network->get_parameters_device() != Device::CPU)
        return;

    const float* parameters_data = neural_network->get_parameters_data();
    const Index parameters_size = neural_network->get_parameters_size();

    for (Index i = 0; i < parameters_size; ++i)
        throw_if(!isfinite(parameters_data[i]),
                 "ModelExpression: network parameters contain NaN or Inf; cannot export a valid model.");
}

string ModelExpression::build_expression() const
{
    auto* network = const_cast<NeuralNetwork*>(neural_network);
    const bool was_on_device = (neural_network->get_parameters_device() == Device::CUDA);
    if (was_on_device) network->copy_parameters_host();

    const size_t layers_number = neural_network->get_layers_number();
    const vector<string> layer_labels = neural_network->get_layer_labels();

    check_parameters_are_finite();

    const Index inputs_number = get_flat_inputs_number();
    const Index outputs_number = neural_network->get_outputs_number();

    vector<string> new_input_names = get_flat_input_names();
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
        const LayerType layer_type = layers[i]->get_type();

        throw_if(layer_type != LayerType::Scaling
                 && layer_type != LayerType::Dense
                 && layer_type != LayerType::Recurrent
                 && layer_type != LayerType::LongShortTermMemory
                 && layer_type != LayerType::Unscaling
                 && layer_type != LayerType::Bounding,
                 format("ModelExpression: layer '{}' ({}) is not supported for export.",
                        layer_labels[i], layer_type_map().to_string(layer_type)));

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

        const string layer_expression = layers[i]->write_expression(new_input_names, layer_output_names);

        throw_if(!is_last && layer_expression.find("Softmax") != string::npos,
                 "ModelExpression: Softmax in a hidden layer is not supported for export.");

        buffer << layer_expression << "\n";

        if (!is_last)
            new_input_names = move(layer_output_names);
    }

    if (was_on_device) network->copy_parameters_device();

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
    const size_t count = min(input_names.size(), fixed_input_names.size());
    for (size_t i = 0; i < count; ++i)
    {
        const string& raw = input_names[i];
        const string& fix = fixed_input_names[i];
        if (raw == fix) continue;
        replace_all_appearances(processed, "scaled_" + raw, "scaled_" + fix);
        string space_to_underscore = raw;
        ranges::replace(space_to_underscore, ' ', '_');
        if (space_to_underscore != raw)
            replace_all_appearances(processed, "scaled_" + space_to_underscore,
                                    "scaled_" + fix);
    }
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
            "// Returns the raw logit: the numerically stable softmax is applied over the whole output vector afterwards.\nfloat Softmax(float x) {\n\treturn x;\n}\n\n",
            "// Returns the raw logit: the numerically stable softmax is applied over the whole output vector afterwards.\nfunction Softmax(x) {\n\treturn x;\n}\n",
            "\t@staticmethod\n\tdef Softmax(x):\n\t\t# Raw logit: the stable softmax is applied over the whole output vector afterwards\n\t\treturn x\n\n",
            "// Returns the raw logit: the numerically stable softmax is applied over the whole output vector afterwards.\nfunction Softmax($x) { return $x; }\n"
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
    const vector<string> input_names = get_flat_input_names();

    buffer << c_header;

    for (size_t i = 0; i < input_names.size(); ++i)
        buffer << "\n// \t " << i << ")  " << input_names[i];

    buffer << "\n \n \n#include <math.h>\n\n"
              "#ifndef OPENNN_EXPORT_NO_MAIN\n"
              "#include <stdio.h>\n"
              "#endif\n\n"
              "static double max(double a, double b) { return a > b ? a : b; }\n"
              "static double min(double a, double b) { return a < b ? a : b; }\n\n";
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
    const vector<string> input_names = get_flat_input_names();
    const vector<string> output_names = neural_network->get_output_feature_names();
    const vector<string> fixed_input_names = fix_names(input_names, "input_");
    const vector<string> fixed_output_names = fix_names(output_names, "output_");

    const Index inputs_number = input_names.size();
    const Index outputs_number = output_names.size();

    buffer << "float* calculate_outputs(const float* inputs)\n{\n";

    for (Index i = 0; i < inputs_number; ++i)
        buffer << "\tconst float " << fixed_input_names[i] << " = inputs[" << i << "];\n";

    buffer << "\n";

    unordered_set<string> declared;
    for (const string& l : lines)
    {
        const string processed = process_body_line(l, input_names, fixed_input_names);
        const size_t eq = processed.find('=');
        const string lhs = eq == string::npos ? "" : get_trimmed(processed.substr(0, eq));
        const bool first = !lhs.empty() && declared.insert(lhs).second;
        buffer << "\t" << (first ? "double " : "") << processed << "\n";
    }

    const vector<string> fixed_outputs = fix_output_names(expression, output_names, ProgrammingLanguage::C);
    if (!fixed_outputs.empty())
    {
        buffer << "\n";
        for (const string& l : fixed_outputs)
            buffer << "\t" << l << "\n";
    }

    buffer << "\n\tstatic float out[" << outputs_number << "];\n\n";

    for (Index i = 0; i < outputs_number; ++i)
        buffer << "\tout[" << i << "] = " << fixed_output_names[i] << ";\n";

    if (has_softmax)
        buffer << "\n\t// Softmax (numerically stable)\n"
                  "\tfloat max_out = out[0];\n"
                  "\tfor(int i = 1; i < " << outputs_number << "; ++i) if(out[i] > max_out) max_out = out[i];\n"
                  "\tfloat sum = 0.0f;\n"
                  "\tfor(int i = 0; i < " << outputs_number << "; ++i) { out[i] = expf(out[i] - max_out); sum += out[i]; }\n"
                  "\tfor(int i = 0; i < " << outputs_number << "; ++i) out[i] /= sum;\n";

    buffer << "\n\treturn out;\n}\n\n";
}

void ModelExpression::emit_c_main(ostringstream& buffer) const
{
    const vector<string> input_names = get_flat_input_names();
    const vector<string> output_names = neural_network->get_output_feature_names();

    const Index inputs_number = input_names.size();
    const Index outputs_number = output_names.size();

    buffer << "#ifndef OPENNN_EXPORT_NO_MAIN\n\n"
              "int main() { \n\n"
              "\tfloat inputs[" << inputs_number << "];\n\n"
              "\t// Please enter your values here:\n";

    for (Index i = 0; i < inputs_number; ++i)
        buffer << "\tinputs[" << i << "] = 0.0f; // " << input_names[i] << "\n";

    buffer << "\n\tconst float* outputs = calculate_outputs(inputs);\n\n"
              "\tprintf(\"These are your outputs:\\n\");\n";

    for (Index i = 0; i < outputs_number; ++i)
        buffer << "\tprintf(\""<< output_names[i] << ": %f \\n\", outputs[" << i << "]);\n";

    buffer << "\n\treturn 0;\n} \n\n"
              "#endif // OPENNN_EXPORT_NO_MAIN\n";
}

string ModelExpression::c_float_literal(float value)
{
    string text = format("{:.9g}", value);

    if (text.find('.') == string::npos && text.find('e') == string::npos)
        text += ".0";

    return text + "f";
}

string ModelExpression::get_expression_c_embedded() const
{
    auto* network = const_cast<NeuralNetwork*>(neural_network);
    const bool was_on_device = (neural_network->get_parameters_device() == Device::CUDA);
    if (was_on_device) network->copy_parameters_host();

    const auto& layers = neural_network->get_layers();
    const size_t layers_number = layers.size();
    const vector<string> layer_labels = neural_network->get_layer_labels();

    throw_if(layers_number == 0, "ModelExpression: the network is empty.");

    check_parameters_are_finite();

    const vector<string> input_names = get_flat_input_names();
    const Index inputs_number = get_flat_inputs_number();
    const Index outputs_number = neural_network->get_outputs_number();

    ostringstream tables;
    ostringstream body;

    bool uses_dense = false;
    bool uses_affine = false;
    bool uses_affine_flags = false;
    bool uses_softmax = false;
    bool uses_clamp = false;
    bool uses_recurrent = false;
    bool uses_lstm = false;
    bool buffer_used[2] = {false, false};

    Index max_width = inputs_number;
    Index max_hidden = 0;
    string current = "inputs";
    const char* buffer_names[2] = {"nn_buffer_a", "nn_buffer_b"};
    int next_buffer = 0;

    auto activation_constant_for = [](ActivationFunction activation) -> string
    {
        using enum ActivationFunction;
        switch (activation)
        {
        case Identity:  return "NN_IDENTITY";
        case Sigmoid:   return "NN_SIGMOID";
        case Tanh:      return "NN_TANH";
        case ReLU:      return "NN_RELU";
        case LeakyReLU: return "NN_LEAKY_RELU";
        default:
            throw runtime_error("ModelExpression: activation function not supported in embedded export.");
        }
    };

    auto emit_float_array = [&](const string& name, const float* values, size_t count)
    {
        tables << "static const float " << name << "[" << count << "] NN_FLASH = {";
        for (size_t k = 0; k < count; ++k)
        {
            if (k % 8 == 0) tables << "\n    ";
            tables << c_float_literal(values[k]);
            if (k + 1 < count) tables << ", ";
        }
        tables << "\n};\n\n";
    };

    auto emit_byte_array = [&](const string& name, const vector<unsigned char>& values)
    {
        tables << "static const unsigned char " << name << "[" << values.size() << "] NN_FLASH = {";
        for (size_t k = 0; k < values.size(); ++k)
        {
            if (k % 16 == 0) tables << "\n    ";
            tables << int(values[k]);
            if (k + 1 < values.size()) tables << ", ";
        }
        tables << "\n};\n\n";
    };

    auto take_buffer = [&]() -> string
    {
        buffer_used[next_buffer] = true;
        const string target = buffer_names[next_buffer];
        next_buffer = 1 - next_buffer;
        return target;
    };

    // In-place stages (clamp) must not modify the caller's inputs.
    auto in_place_target = [&]() -> string
    {
        if (current == "inputs")
        {
            const string target = take_buffer();
            body << "\tfor (int i = 0; i < " << inputs_number << "; ++i) "
                 << target << "[i] = inputs[i];\n";
            current = target;
        }
        return current;
    };

    for (size_t i = 0; i < layers_number; ++i)
    {
        const LayerType layer_type = layers[i]->get_type();
        const string table_prefix = format("nn_layer_{}", i);
        const bool is_last = (i == layers_number - 1);

        if (layer_type == LayerType::Scaling || layer_type == LayerType::Unscaling)
        {
            const Scaling* scaling = static_cast<const Scaling*>(layers[i].get());
            const vector<Descriptives>& descriptives = scaling->get_descriptives();
            const vector<ScalerMethod>& scalers = scaling->get_scalers();
            const float min_range = scaling->get_min_range();
            const float max_range = scaling->get_max_range();
            // Rank-2 (time series) inputs have one scaler per feature applied
            // at every time step: total values = time_steps * features.
            const Index total_number = layers[i]->get_outputs_number();
            const Index features_number = ssize(scalers);
            const bool is_unscaling = (layer_type == LayerType::Unscaling);

            throw_if(features_number == 0
                     || total_number % features_number != 0
                     || ssize(descriptives) != features_number,
                     format("ModelExpression: layer '{}' is not configured.", layer_labels[i]));

            vector<float> slopes(features_number), offsets(features_number);
            vector<unsigned char> log_pre(features_number, 0), exp_post(features_number, 0);
            bool any_flag = false;

            for (Index f = 0; f < features_number; ++f)
            {
                const Descriptives& d = descriptives[size_t(f)];
                float slope = 1.0f;
                float offset = 0.0f;

                using enum ScalerMethod;
                switch (scalers[size_t(f)])
                {
                case None:
                    break;
                case MinimumMaximum:
                    if (is_unscaling)
                    {
                        if (abs(d.maximum - d.minimum) < EPSILON)
                        {
                            slope = 0.0f;
                            offset = d.minimum;
                        }
                        else
                        {
                            slope = (d.maximum - d.minimum) / (max_range - min_range);
                            offset = d.minimum - min_range * slope;
                        }
                    }
                    else
                    {
                        if (d.maximum - d.minimum < EPSILON)
                            slope = 0.0f;
                        else
                        {
                            slope = (max_range - min_range) / (d.maximum - d.minimum);
                            offset = min_range - d.minimum * slope;
                        }
                    }
                    break;
                case MeanStandardDeviation:
                    if (is_unscaling)
                    {
                        slope = d.standard_deviation;
                        offset = d.mean;
                    }
                    else if (d.standard_deviation > EPSILON)
                    {
                        slope = 1.0f / d.standard_deviation;
                        offset = -d.mean / d.standard_deviation;
                    }
                    else
                        slope = 0.0f;
                    break;
                case StandardDeviation:
                    if (is_unscaling)
                        slope = d.standard_deviation;
                    else if (d.standard_deviation > EPSILON)
                        slope = 1.0f / d.standard_deviation;
                    else
                        slope = 0.0f;
                    break;
                case Logarithm:
                    (is_unscaling ? exp_post : log_pre)[size_t(f)] = 1;
                    any_flag = true;
                    break;
                case ImageMinMax:
                    slope = is_unscaling ? 255.0f : 1.0f / 255.0f;
                    break;
                default:
                    throw runtime_error("ModelExpression: unknown scaling method.");
                }

                slopes[size_t(f)] = slope;
                offsets[size_t(f)] = offset;
            }

            emit_float_array(table_prefix + "_a", slopes.data(), slopes.size());
            emit_float_array(table_prefix + "_b", offsets.data(), offsets.size());

            const string target = take_buffer();

            if (any_flag)
            {
                uses_affine_flags = true;
                emit_byte_array(table_prefix + "_log_pre", log_pre);
                emit_byte_array(table_prefix + "_exp_post", exp_post);
                body << "\tnn_affine_flags_forward(" << current << ", " << table_prefix << "_log_pre, "
                     << table_prefix << "_a, " << table_prefix << "_b, " << table_prefix << "_exp_post, "
                     << features_number << ", " << total_number << ", " << target << ");\n";
            }
            else
            {
                uses_affine = true;
                body << "\tnn_affine_forward(" << current << ", " << table_prefix << "_a, "
                     << table_prefix << "_b, " << features_number << ", " << total_number << ", " << target << ");\n";
            }

            current = target;
            max_width = max(max_width, total_number);
        }
        else if (layer_type == LayerType::Dense)
        {
            const Dense* dense = static_cast<const Dense*>(layers[i].get());

            throw_if(dense->get_input_shape().rank != 1,
                     "ModelExpression: only rank-1 Dense inputs are supported in embedded export.");

            throw_if(dense->get_batch_normalization(),
                     "ModelExpression: batch normalization is not supported in the exported model.");

            const vector<TensorView>& parameter_views = dense->get_parameter_views();

            throw_if(parameter_views.size() < 2 || !parameter_views[0].data || !parameter_views[1].data,
                     format("ModelExpression: layer '{}' is not configured.", layer_labels[i]));

            const Index layer_inputs = dense->get_inputs_number();
            const Index layer_outputs = dense->get_outputs_number();

            const float* bias_data = parameter_views[0].as<float>();
            const float* weight_data = parameter_views[1].as<float>();

            emit_float_array(table_prefix + "_weights", weight_data, size_t(layer_inputs * layer_outputs));
            emit_float_array(table_prefix + "_biases", bias_data, size_t(layer_outputs));

            const ActivationFunction activation = dense->get_activation_function();
            const bool is_softmax = (activation == ActivationFunction::Softmax);

            throw_if(is_softmax && !is_last,
                     "ModelExpression: Softmax in a hidden layer is not supported for export.");

            // Softmax emits raw logits (identity) and is normalized over the
            // output vector below.
            const string activation_constant = is_softmax
                ? "NN_IDENTITY"
                : activation_constant_for(activation);

            uses_dense = true;

            const string target = take_buffer();

            body << "\tnn_dense_forward(" << current << ", " << layer_inputs << ", "
                 << table_prefix << "_weights, " << table_prefix << "_biases, "
                 << layer_outputs << ", " << activation_constant << ", " << target << ");\n";

            current = target;

            if (is_softmax)
            {
                uses_softmax = true;
                body << "\tnn_softmax_inplace(" << current << ", " << layer_outputs << ");\n";
            }

            max_width = max(max_width, layer_outputs);
        }
        else if (layer_type == LayerType::Bounding)
        {
            const Bounding* bounding = static_cast<const Bounding*>(layers[i].get());

            if (bounding->get_bounding_method() == Bounding::BoundingMethod::NoBounding)
                continue;

            const Index features_number = layers[i]->get_outputs_number();
            const VectorR lower = bounding->get_lower_bounds();
            const VectorR upper = bounding->get_upper_bounds();

            throw_if(ssize(lower) < features_number || ssize(upper) < features_number,
                     format("ModelExpression: layer '{}' is not configured.", layer_labels[i]));

            emit_float_array(table_prefix + "_lower", lower.data(), size_t(features_number));
            emit_float_array(table_prefix + "_upper", upper.data(), size_t(features_number));

            uses_clamp = true;

            body << "\tnn_clamp_inplace(" << in_place_target() << ", " << table_prefix << "_lower, "
                 << table_prefix << "_upper, " << features_number << ");\n";
        }
        else if (layer_type == LayerType::Recurrent)
        {
            const Recurrent* recurrent = static_cast<const Recurrent*>(layers[i].get());
            const vector<TensorView>& parameter_views = recurrent->get_parameter_views();

            throw_if(parameter_views.size() < 3
                     || !parameter_views[0].data || !parameter_views[1].data || !parameter_views[2].data,
                     format("ModelExpression: layer '{}' is not configured.", layer_labels[i]));

            const Shape input_shape = recurrent->get_input_shape();
            const Shape output_shape = recurrent->get_output_shape();
            const Index time_steps = input_shape[0];
            const Index features = input_shape[1];
            const bool return_sequences = (output_shape.rank == 2);
            const Index hidden = output_shape[output_shape.rank - 1];

            const VectorMap biases_map = parameter_views[0].as_vector();
            const MatrixMap input_w_map = parameter_views[1].as_matrix();
            const MatrixMap recurrent_w_map = parameter_views[2].as_matrix();

            vector<float> recurrent_biases(static_cast<size_t>(hidden));
            vector<float> recurrent_input_weights(static_cast<size_t>(features * hidden));
            vector<float> recurrent_recurrent_weights(static_cast<size_t>(hidden * hidden));

            for (Index j = 0; j < hidden; ++j)
                recurrent_biases[size_t(j)] = biases_map(j);
            for (Index f = 0; f < features; ++f)
                for (Index j = 0; j < hidden; ++j)
                    recurrent_input_weights[size_t(f * hidden + j)] = input_w_map(f, j);
            for (Index p = 0; p < hidden; ++p)
                for (Index j = 0; j < hidden; ++j)
                    recurrent_recurrent_weights[size_t(p * hidden + j)] = recurrent_w_map(p, j);

            emit_float_array(table_prefix + "_biases", recurrent_biases.data(), recurrent_biases.size());
            emit_float_array(table_prefix + "_input_weights", recurrent_input_weights.data(), recurrent_input_weights.size());
            emit_float_array(table_prefix + "_recurrent_weights", recurrent_recurrent_weights.data(), recurrent_recurrent_weights.size());

            const string activation_constant = activation_constant_for(
                ActivationOperator::from_string(recurrent->get_activation_function()));

            uses_recurrent = true;
            max_hidden = max(max_hidden, hidden);

            const string target = take_buffer();

            body << "\tnn_recurrent_forward(" << current << ", " << time_steps << ", " << features << ", "
                 << table_prefix << "_input_weights, " << table_prefix << "_recurrent_weights, "
                 << table_prefix << "_biases, " << hidden << ", " << activation_constant << ", "
                 << (return_sequences ? 1 : 0) << ", nn_state_a, nn_state_b, " << target << ");\n";

            current = target;
            max_width = max(max_width, return_sequences ? time_steps * hidden : hidden);
        }
        else if (layer_type == LayerType::LongShortTermMemory)
        {
            const LongShortTermMemory* lstm = static_cast<const LongShortTermMemory*>(layers[i].get());
            const vector<TensorView>& parameter_views = lstm->get_parameter_views();

            bool configured = parameter_views.size() >= 12;
            for (size_t p = 0; configured && p < 12; ++p)
                configured = parameter_views[p].data != nullptr;

            throw_if(!configured,
                     format("ModelExpression: layer '{}' is not configured.", layer_labels[i]));

            const Index time_steps = lstm->get_time_steps();
            const Index features = lstm->get_input_features();
            const Index hidden = lstm->get_output_features();
            const bool return_sequences = lstm->get_return_sequences();

            // Packed tables, gate order: forget, input, candidate, output
            // (matching the parameter views layout: biases 0-3, W 4-7, U 8-11).
            vector<float> lstm_biases(static_cast<size_t>(4 * hidden));
            vector<float> lstm_input_weights(static_cast<size_t>(4 * features * hidden));
            vector<float> lstm_recurrent_weights(static_cast<size_t>(4 * hidden * hidden));

            for (Index gate = 0; gate < 4; ++gate)
            {
                const VectorMap gate_biases = parameter_views[size_t(gate)].as_vector();
                const MatrixMap gate_w = parameter_views[size_t(4 + gate)].as_matrix();
                const MatrixMap gate_u = parameter_views[size_t(8 + gate)].as_matrix();

                for (Index j = 0; j < hidden; ++j)
                    lstm_biases[size_t(gate * hidden + j)] = gate_biases(j);
                for (Index f = 0; f < features; ++f)
                    for (Index j = 0; j < hidden; ++j)
                        lstm_input_weights[size_t(gate * features * hidden + f * hidden + j)] = gate_w(f, j);
                for (Index p = 0; p < hidden; ++p)
                    for (Index j = 0; j < hidden; ++j)
                        lstm_recurrent_weights[size_t(gate * hidden * hidden + p * hidden + j)] = gate_u(p, j);
            }

            emit_float_array(table_prefix + "_biases", lstm_biases.data(), lstm_biases.size());
            emit_float_array(table_prefix + "_input_weights", lstm_input_weights.data(), lstm_input_weights.size());
            emit_float_array(table_prefix + "_recurrent_weights", lstm_recurrent_weights.data(), lstm_recurrent_weights.size());

            const string activation_constant = activation_constant_for(lstm->get_activation_function());
            const string recurrent_activation_constant = activation_constant_for(lstm->get_recurrent_activation_function());

            uses_lstm = true;
            max_hidden = max(max_hidden, hidden);

            const string target = take_buffer();

            body << "\tnn_lstm_forward(" << current << ", " << time_steps << ", " << features << ", "
                 << table_prefix << "_biases, " << table_prefix << "_input_weights, "
                 << table_prefix << "_recurrent_weights, " << hidden << ", "
                 << activation_constant << ", " << recurrent_activation_constant << ", "
                 << (return_sequences ? 1 : 0) << ", nn_state_a, nn_state_b, nn_cell, " << target << ");\n";

            current = target;
            max_width = max(max_width, return_sequences ? time_steps * hidden : hidden);
        }
        else
        {
            throw runtime_error(format("ModelExpression: layer '{}' ({}) is not supported in embedded export.",
                                       layer_labels[i], layer_type_map().to_string(layer_type)));
        }
    }

    if (was_on_device) network->copy_parameters_device();

    throw_if(current == "inputs", "ModelExpression: the network has no layers to export.");

    ostringstream buffer;

    buffer << c_header;

    for (size_t i = 0; i < input_names.size(); ++i)
        buffer << "\n// \t " << i << ")  " << input_names[i];

    buffer << "\n \n \n#include <math.h>\n\n"
              "#ifndef OPENNN_EXPORT_NO_MAIN\n"
              "#include <stdio.h>\n"
              "#endif\n\n"
              "// On AVR (Harvard architecture) the weight tables are placed in flash\n"
              "// (PROGMEM) and read through pgm_read_*; elsewhere they are plain const data.\n"
              "#ifdef __AVR__\n"
              "#include <avr/pgmspace.h>\n"
              "#define NN_FLASH PROGMEM\n"
              "#define NN_READ_FLOAT(address) pgm_read_float(address)\n"
              "#define NN_READ_BYTE(address) pgm_read_byte(address)\n"
              "#else\n"
              "#define NN_FLASH\n"
              "#define NN_READ_FLOAT(address) (*(address))\n"
              "#define NN_READ_BYTE(address) (*(address))\n"
              "#endif\n\n"
              "#define NN_INPUTS_NUMBER " << inputs_number << "\n"
              "#define NN_OUTPUTS_NUMBER " << outputs_number << "\n"
              "#define NN_MAX_WIDTH " << max_width << "\n\n";

    if (uses_dense || uses_recurrent || uses_lstm)
        buffer << "typedef enum { NN_IDENTITY, NN_SIGMOID, NN_TANH, NN_RELU, NN_LEAKY_RELU } nn_activation;\n\n"
                  "static float nn_activation_forward(nn_activation activation, float x)\n{\n"
                  "\tswitch (activation)\n\t{\n"
                  "\tcase NN_SIGMOID:    return 1.0f / (1.0f + expf(-x));\n"
                  "\tcase NN_TANH:       return tanhf(x);\n"
                  "\tcase NN_RELU:       return x > 0.0f ? x : 0.0f;\n"
                  "\tcase NN_LEAKY_RELU: return x >= 0.0f ? x : x * " << c_float_literal(LEAKY_RELU_SLOPE) << ";\n"
                  "\tdefault:            return x;\n"
                  "\t}\n}\n\n";

    if (uses_dense)
        buffer << "static void nn_dense_forward(const float* inputs, int inputs_number,\n"
                  "                             const float* weights, const float* biases,\n"
                  "                             int outputs_number, nn_activation activation, float* outputs)\n{\n"
                  "\tfor (int j = 0; j < outputs_number; ++j)\n\t{\n"
                  "\t\tfloat sum = NN_READ_FLOAT(&biases[j]);\n"
                  "\t\tfor (int i = 0; i < inputs_number; ++i)\n"
                  "\t\t\tsum += inputs[i] * NN_READ_FLOAT(&weights[i * outputs_number + j]);\n"
                  "\t\toutputs[j] = nn_activation_forward(activation, sum);\n"
                  "\t}\n}\n\n";

    // The affine tables hold one entry per feature; for rank-2 (time series)
    // inputs they are tiled over the time steps (total = time_steps * features).
    if (uses_affine)
        buffer << "static void nn_affine_forward(const float* inputs, const float* a, const float* b,\n"
                  "                              int features, int total, float* outputs)\n{\n"
                  "\tint f = 0;\n"
                  "\tfor (int i = 0; i < total; ++i)\n"
                  "\t{\n"
                  "\t\toutputs[i] = NN_READ_FLOAT(&a[f]) * inputs[i] + NN_READ_FLOAT(&b[f]);\n"
                  "\t\tif (++f == features) f = 0;\n"
                  "\t}\n}\n\n";

    if (uses_affine_flags)
        buffer << "static void nn_affine_flags_forward(const float* inputs, const unsigned char* log_pre,\n"
                  "                                    const float* a, const float* b,\n"
                  "                                    const unsigned char* exp_post, int features, int total,\n"
                  "                                    float* outputs)\n{\n"
                  "\tint f = 0;\n"
                  "\tfor (int i = 0; i < total; ++i)\n"
                  "\t{\n"
                  "\t\tfloat value = NN_READ_BYTE(&log_pre[f]) ? logf(inputs[i]) : inputs[i];\n"
                  "\t\tvalue = NN_READ_FLOAT(&a[f]) * value + NN_READ_FLOAT(&b[f]);\n"
                  "\t\toutputs[i] = NN_READ_BYTE(&exp_post[f]) ? expf(value) : value;\n"
                  "\t\tif (++f == features) f = 0;\n"
                  "\t}\n}\n\n";

    if (uses_recurrent)
        buffer << "static void nn_recurrent_forward(const float* inputs, int time_steps, int input_features,\n"
                  "                                 const float* input_weights, const float* recurrent_weights,\n"
                  "                                 const float* biases, int hidden, nn_activation activation,\n"
                  "                                 int return_sequences, float* state_previous, float* state_current,\n"
                  "                                 float* outputs)\n{\n"
                  "\tfor (int t = 0; t < time_steps; ++t)\n\t{\n"
                  "\t\tconst float* x = inputs + t * input_features;\n"
                  "\t\tfor (int j = 0; j < hidden; ++j)\n\t\t{\n"
                  "\t\t\tfloat sum = NN_READ_FLOAT(&biases[j]);\n"
                  "\t\t\tfor (int i = 0; i < input_features; ++i)\n"
                  "\t\t\t\tsum += x[i] * NN_READ_FLOAT(&input_weights[i * hidden + j]);\n"
                  "\t\t\tif (t > 0)\n"
                  "\t\t\t\tfor (int p = 0; p < hidden; ++p)\n"
                  "\t\t\t\t\tsum += state_previous[p] * NN_READ_FLOAT(&recurrent_weights[p * hidden + j]);\n"
                  "\t\t\tstate_current[j] = nn_activation_forward(activation, sum);\n"
                  "\t\t}\n"
                  "\t\tif (return_sequences)\n"
                  "\t\t\tfor (int j = 0; j < hidden; ++j) outputs[t * hidden + j] = state_current[j];\n"
                  "\t\t{ float* swap = state_previous; state_previous = state_current; state_current = swap; }\n"
                  "\t}\n"
                  "\tif (!return_sequences)\n"
                  "\t\tfor (int j = 0; j < hidden; ++j) outputs[j] = state_previous[j];\n"
                  "}\n\n";

    if (uses_lstm)
        buffer << "// Packed tables, gate order: forget, input, candidate, output.\n"
                  "// c_t = f * c_(t-1) + i * g ; h_t = o * activation(c_t) ; h_(-1) = c_(-1) = 0.\n"
                  "static void nn_lstm_forward(const float* inputs, int time_steps, int input_features,\n"
                  "                            const float* biases, const float* input_weights,\n"
                  "                            const float* recurrent_weights, int hidden,\n"
                  "                            nn_activation activation, nn_activation recurrent_activation,\n"
                  "                            int return_sequences, float* state_previous, float* state_current,\n"
                  "                            float* cell_state, float* outputs)\n{\n"
                  "\tfor (int t = 0; t < time_steps; ++t)\n\t{\n"
                  "\t\tconst float* x = inputs + t * input_features;\n"
                  "\t\tfor (int j = 0; j < hidden; ++j)\n\t\t{\n"
                  "\t\t\tfloat gates[4];\n"
                  "\t\t\tfor (int gate = 0; gate < 4; ++gate)\n\t\t\t{\n"
                  "\t\t\t\tfloat sum = NN_READ_FLOAT(&biases[gate * hidden + j]);\n"
                  "\t\t\t\tconst float* w = input_weights + gate * input_features * hidden;\n"
                  "\t\t\t\tfor (int i = 0; i < input_features; ++i)\n"
                  "\t\t\t\t\tsum += x[i] * NN_READ_FLOAT(&w[i * hidden + j]);\n"
                  "\t\t\t\tif (t > 0)\n\t\t\t\t{\n"
                  "\t\t\t\t\tconst float* u = recurrent_weights + gate * hidden * hidden;\n"
                  "\t\t\t\t\tfor (int p = 0; p < hidden; ++p)\n"
                  "\t\t\t\t\t\tsum += state_previous[p] * NN_READ_FLOAT(&u[p * hidden + j]);\n"
                  "\t\t\t\t}\n"
                  "\t\t\t\tgates[gate] = nn_activation_forward(gate == 2 ? activation : recurrent_activation, sum);\n"
                  "\t\t\t}\n"
                  "\t\t\tcell_state[j] = (t > 0 ? gates[0] * cell_state[j] : 0.0f) + gates[1] * gates[2];\n"
                  "\t\t\tstate_current[j] = gates[3] * nn_activation_forward(activation, cell_state[j]);\n"
                  "\t\t}\n"
                  "\t\tif (return_sequences)\n"
                  "\t\t\tfor (int j = 0; j < hidden; ++j) outputs[t * hidden + j] = state_current[j];\n"
                  "\t\t{ float* swap = state_previous; state_previous = state_current; state_current = swap; }\n"
                  "\t}\n"
                  "\tif (!return_sequences)\n"
                  "\t\tfor (int j = 0; j < hidden; ++j) outputs[j] = state_previous[j];\n"
                  "}\n\n";

    if (uses_softmax)
        buffer << "static void nn_softmax_inplace(float* values, int n)\n{\n"
                  "\tfloat max_value = values[0];\n"
                  "\tfor (int i = 1; i < n; ++i) if (values[i] > max_value) max_value = values[i];\n"
                  "\tfloat sum = 0.0f;\n"
                  "\tfor (int i = 0; i < n; ++i) { values[i] = expf(values[i] - max_value); sum += values[i]; }\n"
                  "\tfor (int i = 0; i < n; ++i) values[i] /= sum;\n"
                  "}\n\n";

    if (uses_clamp)
        buffer << "static void nn_clamp_inplace(float* values, const float* lower, const float* upper, int n)\n{\n"
                  "\tfor (int i = 0; i < n; ++i)\n"
                  "\t{\n"
                  "\t\tconst float low = NN_READ_FLOAT(&lower[i]);\n"
                  "\t\tconst float high = NN_READ_FLOAT(&upper[i]);\n"
                  "\t\tif (values[i] < low) values[i] = low;\n"
                  "\t\tif (values[i] > high) values[i] = high;\n"
                  "\t}\n}\n\n";

    buffer << tables.str();

    buffer << "float* calculate_outputs(const float* inputs)\n{\n";

    if (buffer_used[0])
        buffer << "\tstatic float nn_buffer_a[NN_MAX_WIDTH];\n";
    if (buffer_used[1])
        buffer << "\tstatic float nn_buffer_b[NN_MAX_WIDTH];\n";

    if (uses_recurrent || uses_lstm)
    {
        buffer << "\tstatic float nn_state_a[" << max_hidden << "];\n"
                  "\tstatic float nn_state_b[" << max_hidden << "];\n";
        if (uses_lstm)
            buffer << "\tstatic float nn_cell[" << max_hidden << "];\n";
    }

    buffer << "\n" << body.str()
           << "\n\treturn " << current << ";\n}\n\n";

    emit_c_main(buffer);

    return buffer.str();
}

string ModelExpression::get_expression_php() const
{
    const vector<string> input_names = get_flat_input_names();
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
    const vector<string> input_names = get_flat_input_names();

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
    const vector<string> fixed_input_names = fix_names(get_flat_input_names(), "input_");

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
        buffer << "\n// Softmax (numerically stable)\n$max_out = max(";
        for (size_t i = 0; i < fixed_output_names.size(); ++i)
            buffer << (i ? ", " : "") << "$" << fixed_output_names[i];
        buffer << ");\n$sum = 0;\n";
        for (size_t i = 0; i < fixed_output_names.size(); ++i)
            buffer << "$" << fixed_output_names[i] << " = exp($" << fixed_output_names[i] << " - $max_out);\n"
                   << "$sum += $" << fixed_output_names[i] << ";\n";
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
    const vector<string> input_names = get_flat_input_names();

    buffer << javascript_header;
    for (size_t i = 0; i < input_names.size(); ++i)
        buffer << "\n\t " << i + 1 << ")  " << input_names[i];
    buffer << javascript_subheader;
}

void ModelExpression::emit_js_inputs_html(ostringstream& buffer) const
{
    const vector<string> input_names = get_flat_input_names();
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
    const vector<string> input_names = get_flat_input_names();
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
        if (contains({"Identity", "Tanh"}, name) || expression.find(name) != string::npos)
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

        for (const char* kw : math_keywords)
            replace_all_word_appearances(line, kw, string("Math.") + kw);

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
        buffer << "\n\t// Softmax (numerically stable)\n"
                  "\tvar max_out = out[0];\n"
                  "\tfor(var i = 1; i < out.length; ++i) if(out[i] > max_out) max_out = out[i];\n"
                  "\tvar sum = 0;\n"
                  "\tfor(var i = 0; i < out.length; ++i) { out[i] = Math.exp(out[i] - max_out); sum += out[i]; }\n"
                  "\tfor(var i = 0; i < out.length; ++i) out[i] /= sum;\n";

    buffer << "\n\treturn out;\n}\n\n"
              "function updateTextInput1(value, id)\n{\n"
              "\tdocument.getElementById(id).value = value;\n"
              "}\n\n"
              "</script>\n\n"
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
    emit_python_calculate_outputs(buffer, expression, lines, has_softmax);
    emit_python_batch_and_main(buffer);

    string out = buffer.str();
    replace(out, ";", "");
    return out;
}

void ModelExpression::emit_python_prelude(ostringstream& buffer) const
{
    const vector<string> input_names = get_flat_input_names();

    buffer << python_header;
    for (size_t i = 0; i < input_names.size(); ++i)
        buffer << "\t" << i << ") " << input_names[i] << "\n";
    buffer << python_subheader;
}

void ModelExpression::emit_python_class_header(ostringstream& buffer) const
{
    const vector<string> input_names = get_flat_input_names();

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
                                                    const string& expression,
                                                    const vector<string>& lines,
                                                    bool has_softmax) const
{
    const vector<string> input_names = get_flat_input_names();
    const vector<string> output_names = neural_network->get_output_feature_names();
    const vector<string> fixed_output_names = fix_names(output_names, "output_");

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

    const vector<string> output_fixups =
        fix_output_names(expression, output_names, ProgrammingLanguage::Python);
    for (const string& l : output_fixups)
        buffer << "\t\t" << l << "\n";

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
        buffer << "\t\t# Softmax (numerically stable)\n";
        buffer << "\t\tmax_out = np.max(outputs)\n";
        buffer << "\t\toutputs = [np.exp(x - max_out) for x in outputs]\n";
        buffer << "\t\tsum_val = np.sum(outputs)\n";
        buffer << "\t\toutputs = [x / sum_val for x in outputs]\n";
    }
    buffer << "\t\treturn outputs\n\n";
}

void ModelExpression::emit_python_batch_and_main(ostringstream& buffer) const
{
    const vector<string> input_names = get_flat_input_names();
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
        {"min", "mi_n"}, {"max", "ma_x"}, {"exp", "ex_p"}, {"tanh", "ta_nh"},
        {"yield", "yield_"}, {"class", "class_"}, {"return", "return_"},
        {"lambda", "lambda_"}, {"global", "global_"}
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
        replace_all_word_appearances(out, search, replace_val);

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
    tokens.reserve(num_outputs);

    while (getline(ss, token, '\n'))
    {
        if (token.empty() || ranges::all_of(token, [](char c) { return isspace(c); }))
            continue;

        if (token.find('{') != string::npos)
            break;

        if (token.find('=') != string::npos)
            tokens.push_back(token);
    }

    if (tokens.size() < num_outputs)
        return {};

    const vector<string> fixed_outputs = fix_names(outputs, "output_");
    out.reserve(num_outputs);

    const char* lhs_prefix = "";
    const char* rhs_prefix = "";
    const char* suffix = "";

    using enum ProgrammingLanguage;
    switch (programming_language)
    {
    case C:
        lhs_prefix = "double ";
        suffix = ";";
        break;
    case JavaScript:
        lhs_prefix = "\tvar ";
        suffix = ";";
        break;
    case Python:
        break;
    case PHP:
        lhs_prefix = "$";
        rhs_prefix = "$";
        suffix = ";";
        break;
    }

    for (size_t i = 0; i < num_outputs; ++i)
    {
        const string intermediate_var_name = get_first_word(tokens[tokens.size() - num_outputs + i]);
        const string& final_output_name = fixed_outputs[i];

        if (final_output_name != intermediate_var_name)
            out.push_back(string(lhs_prefix) + final_output_name + " = " + rhs_prefix + intermediate_var_name + suffix);
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

    throw_if(!file.is_open(),
             format("Cannot open file: {}", file_name.string()));

    using enum ProgrammingLanguage;
    switch (language)
    {
    case C:          file << get_expression_c();          break;
    case CEmbedded:  file << get_expression_c_embedded(); break;
    case Python:     file << get_expression_python();     break;
    case JavaScript: file << get_expression_javascript(); break;
    case PHP:        file << get_expression_php();        break;
    }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
